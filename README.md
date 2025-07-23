# UCL Scheduler

An automated availability-based rehearsal scheduler that optimizes rehearsal schedules using constraint programming and optimization algorithms.

## Features

- **Availability-based scheduling**: Uses Google Sheets to import cast member availability
- **Multiple leaders support**: Schedule rehearsals with different leaders
- **Optimization**: Find the best schedule based on time preferences, room preferences, and continuity
- **Web interface**: Easy-to-use web form for creating rehearsal requests
- **Real-time scheduling**: Generate optimal schedules instantly

## Deployment

This app is configured for Railway deployment. The main entry point is:

```
python -m app
```

## Requirements

- Python 3.11+
- Google Sheets API credentials
- All dependencies listed in `requirements.txt`

## Local Development

1. Install dependencies: `pip install -r requirements.txt`
2. Set up Google Sheets API credentials
3. Run: `python -m app`
4. Visit: `http://localhost:8080`