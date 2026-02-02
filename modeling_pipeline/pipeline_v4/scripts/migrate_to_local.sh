#!/bin/bash
# Migrate data from Supabase to local PostgreSQL

echo "=========================================="
echo "MIGRATING DATA FROM SUPABASE TO LOCAL DB"
echo "=========================================="

# Supabase connection (update if needed)
SUPABASE_HOST="db.kllrcahfxspfpzmxzwgt.supabase.co"
SUPABASE_DB="postgres"
SUPABASE_USER="postgres.kllrcahfxspfpzmxzwgt"

# Local connection
LOCAL_DB="football_predictions"
LOCAL_USER="ankurgupta"

echo ""
echo "Exporting predictions from Supabase..."
PGPASSWORD="Danbrown1989!" pg_dump \
  -h "$SUPABASE_HOST" \
  -U "$SUPABASE_USER" \
  -d "$SUPABASE_DB" \
  -t predictions \
  --data-only \
  --column-inserts \
  > /tmp/predictions_backup.sql

if [ $? -eq 0 ]; then
    echo "✅ Export successful"
    echo ""
    echo "Importing to local database..."
    psql -d "$LOCAL_DB" -f /tmp/predictions_backup.sql

    if [ $? -eq 0 ]; then
        echo "✅ Import successful"
        echo ""
        echo "Verifying data..."
        psql -d "$LOCAL_DB" -c "SELECT COUNT(*) as total_predictions FROM predictions;"
        rm /tmp/predictions_backup.sql
    else
        echo "❌ Import failed"
        echo "Backup saved at: /tmp/predictions_backup.sql"
    fi
else
    echo "❌ Export failed - check Supabase connection"
    echo ""
    echo "Alternative: Export manually from Supabase SQL Editor:"
    echo "  SELECT * FROM predictions"
    echo "  Then click 'Download as CSV'"
fi
