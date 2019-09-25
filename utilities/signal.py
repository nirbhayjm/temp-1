import os
import signal
import time
from time import gmtime, strftime


class GracefulSignalHandler:
    '''
    Adapted from:
    stackoverflow.com/questions/18499497/how-to-process-sigterm-signal-gracefully
    '''
    def __init__(self):
        # SIGINT and SIGTERM
        # signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

        # User defined signals
        signal.signal(signal.SIGUSR1, self.sig_usr1)

        self.parent_pid = os.getpid()
        # self.info_save_path = info_save_path
        self.kill_now = False

    def exit_gracefully(self, signal, frame):
        if os.getpid() == self.parent_pid:
            self._exit_time = self.get_time_str()
            self.signal_caught = "{}".format(signal)
            print("Exit signal registered: {}".format(signal))
            print("Time when signal reveived: {}".format(self._exit_time))
            self.kill_now = True

    def sig_usr1(self, signal, frame):
        if os.getpid() == self.parent_pid:
            print("Caught user defined signal: {}".format(signal))
            pass

    def get_time_str(self):
        return strftime('%d-%b-%y-%X', gmtime())

    ''' Old, not so graceful interrupt handler below:'''
    # def interrupt_handler(self, signal, frame):
    #     timeStamp = strftime('%d-%b-%y-%X-%a', gmtime())
    #     print("Signal received:",signal)
    #     print("Interrupted at %s, ending process..."%timeStamp)
    #     if os.getpid() == self.parent_pid:
    #         # This makes sure visdom doesn't get spammed with N
    #         # text messages where N = num_workers in Dataloader
    #         viz.viz.text("Interrupted at %s"%timeStamp)
    #     sys.exit(0)
