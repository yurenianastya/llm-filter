import os
import sys

from rabbitmq import initialize

def main():
    initialize()

if __name__ == '__main__':
    try:
        print('Initializing worker...')
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)