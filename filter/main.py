from rabbitmq import initialize

if __name__ == '__main__':
    try:
        print('Initializing worker...')
        initialize()
    except KeyboardInterrupt:
        print('Interrupted')
        exit(0)