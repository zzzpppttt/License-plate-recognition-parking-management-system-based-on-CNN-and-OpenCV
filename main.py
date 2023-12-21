# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import subprocess


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def open_folder():
    folder_path = 'pictures'
    subprocess.Popen(f'explorer "{folder_path}"')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    open_folder()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
