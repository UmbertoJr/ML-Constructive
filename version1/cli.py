import argparse
from version1 import Settings, operations, print_sett


def run_op(s):
    print("\n###################\n\n")
    operations[s.operation](s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="the cli.py handles the actions required to replicate the "
                                                 "experiments from the command line")
    parser.add_argument('--operation', type=str, help='choose which operation to call?\t'
                                                      'possible operations are: {create_instances, statistical_study,'
                                                      ' train, solve_TSPLIB}',
                        default='None')

    operation = parser.parse_args()
    settings = Settings()
    if operation.operation == 'create_instances':
        settings.operation = "create training instances"
        print_sett(settings)
        print('\nstarting to create training data')
        run_op(settings)
        print("Training data-set created!")

        settings.operation = "create evaluation instances"
        print('\nstarting to create evaluation data')
        run_op(settings)
        print("Evaluation data-set created!")

    elif operation.operation == "statistical_study":
        settings.operation = "statistical study"
        print_sett(settings)
        run_op(settings)
        print('\nstatistical study ready!')

    elif operation.operation == "train":
        settings.operation = "train"
        print_sett(settings)
        run_op(settings)
        print('\ntraining done!')
        settings.operation = "show results train"
        run_op(settings)

    elif operation.operation == "solve_TSPLIB":
        settings.operation = "test on TSPLIB"
        print_sett(settings)
        run_op(settings)
        print("\n test completed!")

    else:
        print(f"{operation.operation} operation not available!")
