#!/bin/bash

# Check if the user provided the configuration file and name of screen session
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <config_file> <screen_session_name_prefix>"
  exit 1
fi

# Configuration file containing parameters for each session
CONFIG_FILE=$1

# Screen session name prefix
SESSION_NAME_PREFIX=$2

# Maximum number of concurrent sessions
MAX_CONCURRENT_SESSIONS=5

# Kill any existing screen sessions with the same name prefix
for SESSION_NAME in $(screen -list | grep "$SESSION_NAME_PREFIX" | awk '{print $1}'); do
  echo "Killing existing screen session: $SESSION_NAME"
  screen -S "$SESSION_NAME" -X quit
done

# Python script to execute
PYTHON_SCRIPT="/home/kamil/Documents/RLMAPF2/train.py"

# Read the configuration file and start sessions with concurrency
SESSION_INDEX=1
ACTIVE_SESSIONS=0
while IFS= read -r LINE; do
  # Skip empty lines and comments
  if [[ -z "$LINE" || "$LINE" =~ ^# ]]; then
    continue
  fi

  SESSION_NAME="${SESSION_NAME_PREFIX}_${SESSION_INDEX}"
  echo "Starting screen session: $SESSION_NAME with parameters: $LINE"

  # Start the screen session
  screen -dmS "$SESSION_NAME" bash -c "source /home/kamil/Documents/RLMAPF2/.venv/bin/activate && python $PYTHON_SCRIPT $LINE || exit 1"
  ACTIVE_SESSIONS=$((ACTIVE_SESSIONS + 1))

  # Wait if the maximum number of concurrent sessions is reached
  while [ "$ACTIVE_SESSIONS" -ge "$MAX_CONCURRENT_SESSIONS" ]; do
    sleep 5
    # Check for finished sessions
    for ((i = 1; i <= SESSION_INDEX; i++)); do
      SESSION_NAME_CHECK="${SESSION_NAME_PREFIX}_${i}"
      if ! screen -list | grep -q "$SESSION_NAME_CHECK"; then
        ACTIVE_SESSIONS=$((ACTIVE_SESSIONS - 1))
      fi
    done
  done

  SESSION_INDEX=$((SESSION_INDEX + 1))
done < "$CONFIG_FILE"

# Wait for all remaining sessions to finish
while [ "$ACTIVE_SESSIONS" -gt 0 ]; do
  sleep 5
  for ((i = 1; i <= SESSION_INDEX; i++)); do
    SESSION_NAME_CHECK="${SESSION_NAME_PREFIX}_${i}"
    if ! screen -list | grep -q "$SESSION_NAME_CHECK"; then
      ACTIVE_SESSIONS=$((ACTIVE_SESSIONS - 1))
    fi
  done
done

echo "All sessions completed."
