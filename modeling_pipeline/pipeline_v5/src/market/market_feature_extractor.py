"""
Market Feature Extractor
========================

Extracts ~30 market features from odds data for the bet selector model.
Handles both historical JSON and live API formats.

Key: All features use only PRE-MATCH odds (filters by latest_bookmaker_update < starting_at).
"""

from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


# Sharp bookmakers (Pinnacle-like: lower margins, move first)
SHARP_BOOKMAKERS = {2, 5, 34}
# Soft/recreational bookmakers
SOFT_BOOKMAKERS = {1, 3, 16, 20, 23, 29, 35}

# Label normalization: historical JSON uses mixed formats
LABEL_MAP = {
    'home': 'home', '1': 'home', 'h': 'home',
    'draw': 'draw', 'x': 'draw', 'd': 'draw',
    'away': 'away', '2': 'away', 'a': 'away',
}


class MarketFeatureExtractor:
    """Extracts market features from odds data."""

    def extract_from_json(self, fixture: dict) -> Optional[Dict]:
        """Extract features from historical JSON fixture (with pre-match filtering).

        Args:
            fixture: Full fixture dict from historical JSON with 'odds' and 'starting_at'.

        Returns:
            Dict of market features, or None if insufficient odds data.
        """
        odds_list = fixture.get('odds', [])
        if not odds_list:
            return None

        starting_at = fixture.get('starting_at')
        if not starting_at:
            return None

        match_start = datetime.strptime(str(starting_at).strip(), '%Y-%m-%d %H:%M:%S')

        # Filter to pre-match odds only
        pre_match_odds = []
        for odd in odds_list:
            updated = odd.get('latest_bookmaker_update')
            if not updated:
                continue
            try:
                update_time = datetime.strptime(str(updated).strip(), '%Y-%m-%d %H:%M:%S')
            except (ValueError, TypeError):
                continue
            if update_time < match_start:
                pre_match_odds.append(odd)

        return self._extract_features(pre_match_odds)

    def extract_from_api(self, odds_list: list, model_probs: dict = None) -> Optional[Dict]:
        """Extract features from live API odds (pre-match by definition).

        Args:
            odds_list: List of odds dicts from SportMonks API.
            model_probs: Optional dict with 'home', 'draw', 'away' model probabilities.

        Returns:
            Dict of market features, or None if insufficient odds data.
        """
        features = self._extract_features(odds_list)
        if features is None:
            return None

        # Add model-vs-market edge features if probs provided
        if model_probs:
            features = self._add_edge_features(features, model_probs)

        return features

    def build_full_features(self, market_features: dict, model_probs: dict) -> dict:
        """Combine market features with model probabilities into full feature vector.

        Args:
            market_features: Output from extract_from_json or extract_from_api.
            model_probs: Dict with 'home', 'draw', 'away' probabilities.

        Returns:
            Complete feature dict ready for bet selector.
        """
        features = dict(market_features)

        # Model probability features
        features['pred_home_prob'] = model_probs.get('home', 0)
        features['pred_draw_prob'] = model_probs.get('draw', 0)
        features['pred_away_prob'] = model_probs.get('away', 0)
        features['pred_max_prob'] = max(
            model_probs.get('home', 0),
            model_probs.get('draw', 0),
            model_probs.get('away', 0)
        )

        # Predicted outcome encoded (H=0, D=1, A=2)
        probs = {'home': model_probs.get('home', 0),
                 'draw': model_probs.get('draw', 0),
                 'away': model_probs.get('away', 0)}
        pred_outcome = max(probs, key=probs.get)
        features['pred_outcome_encoded'] = {'home': 0, 'draw': 1, 'away': 2}[pred_outcome]

        # Model-vs-market edge
        features = self._add_edge_features(features, model_probs)

        return features

    def _add_edge_features(self, features: dict, model_probs: dict) -> dict:
        """Add model-vs-market edge features."""
        for outcome in ['home', 'draw', 'away']:
            market_prob = features.get(f'{outcome}_market_prob', 0) or 0
            model_prob = model_probs.get(outcome, 0)
            features[f'{outcome}_edge'] = model_prob - market_prob

        # Edge on predicted outcome
        probs = {k: model_probs.get(k, 0) for k in ['home', 'draw', 'away']}
        pred_outcome = max(probs, key=probs.get)
        features['pred_outcome_edge'] = features.get(f'{pred_outcome}_edge', 0)

        return features

    def _extract_features(self, odds_list: list) -> Optional[Dict]:
        """Core feature extraction from a list of odds entries."""
        # Separate by market
        odds_1x2 = self._filter_1x2(odds_list)
        if not any(odds_1x2.values()):
            return None

        features = {}

        # === Category 1: Consensus odds (6 features) ===
        for outcome in ['home', 'draw', 'away']:
            entries = odds_1x2[outcome]
            if entries:
                values = [e['odds'] for e in entries]
                features[f'{outcome}_best_odds'] = max(values)
                features[f'{outcome}_avg_odds'] = np.mean(values)
            else:
                features[f'{outcome}_best_odds'] = 0
                features[f'{outcome}_avg_odds'] = 0

        # === Category 2: Implied probability (4 features) ===
        implied = {}
        for outcome in ['home', 'draw', 'away']:
            best = features[f'{outcome}_best_odds']
            implied[outcome] = (1 / best) if best > 0 else 0

        total_prob = sum(implied.values())
        features['market_overround'] = total_prob

        for outcome in ['home', 'draw', 'away']:
            features[f'{outcome}_market_prob'] = (implied[outcome] / total_prob) if total_prob > 0 else 0

        # === Category 3: Market structure (4 features) ===
        bookmaker_ids = set()
        for outcome in ['home', 'draw', 'away']:
            entries = odds_1x2[outcome]
            for e in entries:
                bookmaker_ids.add(e['bookmaker_id'])
            if entries:
                values = [e['odds'] for e in entries]
                features[f'{outcome}_odds_spread'] = max(values) - min(values)
            else:
                features[f'{outcome}_odds_spread'] = 0

        features['num_bookmakers'] = len(bookmaker_ids)

        # === Category 4: Sharp vs soft spread (3 features) ===
        for outcome in ['home', 'draw', 'away']:
            entries = odds_1x2[outcome]
            sharp = [e['odds'] for e in entries if e['bookmaker_id'] in SHARP_BOOKMAKERS]
            soft = [e['odds'] for e in entries if e['bookmaker_id'] in SOFT_BOOKMAKERS]
            if sharp and soft:
                features[f'{outcome}_sharp_soft_spread'] = np.mean(sharp) - np.mean(soft)
            else:
                features[f'{outcome}_sharp_soft_spread'] = 0

        # === Category 5: Cross-market features (2 features) ===
        ah = self._extract_asian_handicap(odds_list)
        features['ah_main_line'] = ah['main_line'] if ah['main_line'] is not None else 0

        ou = self._extract_over_under(odds_list)
        features['ou_25_best_over'] = ou['over_best'] if ou['over_best'] is not None else 0

        return features

    def _filter_1x2(self, odds_list: list) -> Dict[str, list]:
        """Filter and group 1X2 (Fulltime Result) odds by outcome."""
        result = {'home': [], 'draw': [], 'away': []}

        for odd in odds_list:
            if odd.get('market_id') != 1:
                # Also check market_description for older data
                desc = (odd.get('market_description') or '').lower()
                if desc not in ('fulltime result', 'match winner', '1x2'):
                    continue

            label = (odd.get('label') or '').lower()
            normalized = LABEL_MAP.get(label)
            if not normalized:
                continue

            value_str = str(odd.get('value', 0)).replace(',', '')
            try:
                value = float(value_str)
            except (ValueError, TypeError):
                continue
            if value <= 1:
                continue

            result[normalized].append({
                'odds': value,
                'bookmaker_id': odd.get('bookmaker_id'),
            })

        return result

    def _extract_asian_handicap(self, odds_list: list) -> dict:
        """Extract Asian Handicap main line."""
        ah_odds = defaultdict(list)

        for odd in odds_list:
            if odd.get('market_id') != 6:
                desc = (odd.get('market_description') or '')
                if 'Asian Handicap' not in desc or 'Half' in desc or 'Alternative' in desc:
                    continue

            handicap = odd.get('handicap')
            if handicap is None:
                continue
            try:
                handicap = float(handicap)
            except (ValueError, TypeError):
                continue

            label = (odd.get('label') or '').lower()
            side = LABEL_MAP.get(label)
            if side not in ('home', 'away'):
                continue

            value_str = str(odd.get('value', 0)).replace(',', '')
            try:
                value = float(value_str)
            except (ValueError, TypeError):
                continue
            if value <= 1:
                continue

            ah_odds[handicap].append({'side': side, 'odds': value})

        if not ah_odds:
            return {'main_line': None}

        # Find most balanced line
        best_line = None
        best_balance = float('inf')
        for line, entries in ah_odds.items():
            home = [e['odds'] for e in entries if e['side'] == 'home']
            away = [e['odds'] for e in entries if e['side'] == 'away']
            if home and away:
                balance = abs(np.mean(home) - np.mean(away))
                if balance < best_balance:
                    best_balance = balance
                    best_line = line

        return {'main_line': best_line}

    def _extract_over_under(self, odds_list: list) -> dict:
        """Extract Over/Under 2.5 goals best odds."""
        over_odds = []

        for odd in odds_list:
            if odd.get('market_id') != 80:
                desc = (odd.get('market_description') or '')
                if desc != 'Goals Over/Under':
                    continue

            total = odd.get('total')
            if total is None:
                total = odd.get('handicap')
            try:
                total = float(total)
            except (ValueError, TypeError):
                continue
            if total != 2.5:
                continue

            label = (odd.get('label') or '').lower()
            if 'over' not in label:
                continue

            value_str = str(odd.get('value', 0)).replace(',', '')
            try:
                value = float(value_str)
            except (ValueError, TypeError):
                continue
            if value <= 1:
                continue

            over_odds.append(value)

        return {'over_best': max(over_odds) if over_odds else None}

    @staticmethod
    def get_feature_names() -> list:
        """Return ordered list of all market feature names."""
        names = []

        # Model probability features
        names.extend(['pred_home_prob', 'pred_draw_prob', 'pred_away_prob',
                      'pred_max_prob', 'pred_outcome_encoded'])

        # Consensus odds
        for o in ['home', 'draw', 'away']:
            names.extend([f'{o}_best_odds', f'{o}_avg_odds'])

        # Implied probability
        names.append('market_overround')
        for o in ['home', 'draw', 'away']:
            names.append(f'{o}_market_prob')

        # Model-vs-market edge
        for o in ['home', 'draw', 'away']:
            names.append(f'{o}_edge')
        names.append('pred_outcome_edge')

        # Market structure
        for o in ['home', 'draw', 'away']:
            names.append(f'{o}_odds_spread')
        names.append('num_bookmakers')

        # Sharp vs soft
        for o in ['home', 'draw', 'away']:
            names.append(f'{o}_sharp_soft_spread')

        # Cross-market
        names.extend(['ah_main_line', 'ou_25_best_over'])

        return names
