import argparse
from version1 import Settings, operations, print_sett


def run_op(s):
    print("\n###################\n\n")
    operations[s.operation](s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--operation', type=str, help='which operation to deal', default='test images')

    operation = parser.parse_args()
    settings = Settings()
    if operation.operation == 'create_instances':
        print_sett(settings)
        print('\nstarting to create training data')
        run_op(settings)

        print("Training data created!")
        settings.operation = "create statistical study instances"
        print('\nstarting to create evaluation data')
        run_op(settings)

    elif operation.operation == "statistical study":
        settings.operation = "statistical study"
        print_sett(settings)
        print('\nstatistical study')
        run_op(settings)

    else:
        print("operation not available!")
