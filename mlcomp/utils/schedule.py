import atexit
import logging

from apscheduler.schedulers.background import BackgroundScheduler


def start_schedule(jobs):
    scheduler = BackgroundScheduler()
    for func, interval in jobs:
        scheduler.add_job(func=func, trigger="interval", seconds=interval,
                          max_instances=1)
    scheduler.start()

    # Shut down the scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())

    class NoRunningFilter(logging.Filter):
        def filter(self, record):
            return 'ran tasks' not in str(record.msg)

    for k in logging.root.manager.loggerDict:
        if 'apscheduler' in k:
            logging.getLogger(k).setLevel(logging.ERROR)
        if 'mlcomp' in k:
            logging.getLogger(k).addFilter(NoRunningFilter())
