import atexit

from apscheduler.schedulers.background import BackgroundScheduler


def start_schedule(jobs):
    scheduler = BackgroundScheduler()
    for func, interval in jobs:
        scheduler.add_job(func=func, trigger='interval', seconds=interval,
                          max_instances=1)
    scheduler.start()

    # Shut down the scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())
