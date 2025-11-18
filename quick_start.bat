@echo off
echo ========================================
echo Starting GA Comparison Experiment
echo ========================================
echo.
echo This will run the experiment in this window.
echo You can see the progress in real-time.
echo Press Ctrl+C to stop (checkpoint will save progress).
echo.
pause
python main.py --config config.yaml
pause

