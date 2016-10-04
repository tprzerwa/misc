import datetime
import os
import json
import re
import shutil
from argparse import ArgumentParser


TOTAL_EXCLUDE = [r'.*versioner\.py', r'.*last_source\.json', r'^RELEASE_.{1,}_\d{1,}_\d{1,}_\d{1,}_\d{1,}.*']
EXCLUDE = [r'.*__init__\.py', r'.*__pycache__.*'] + TOTAL_EXCLUDE
VERSION_PATTERNS = [r'.*# __version__ = VERSION_TO_ADD.*', r'# __version__ = VERSION_TO_ADD']
JSON_NAME = '__last_project_version.json'


class Versioner:
    def __init__(self, release_type='build', release_path=None, exclude_tests=True, exclude_patterns=True, enforce_new_version=False, archive=False, message=None):
        self.__project_path = os.path.dirname(os.path.realpath(__file__))
        self.__project_name = self.__project_path.split('\\')[-1]
        self.__source_dict = None
        self.__old_file = None
        self.__message = message
        self.__changed_files = []

        self.__release_types = {'major': 0, 'minor': 1, 'build': 2, 'revision': 3}
        self.__release_type = release_type
        assert release_type in self.__release_types

        self.__release_path = release_path if release_path is not None else os.getcwd()

        self.__exclude_tests = exclude_tests
        self.__exclude_patterns = exclude_patterns
        self.__enforce = enforce_new_version
        self.__archive = archive

        self.__major, self.__minor, self.__build, self.__revision = 0, 0, 1, 0

    def version(self):
        files = self.__get_python_files()
        self.__source_dict = {file: self.__parse_source(file) for file in files}
        if self.__exclude_tests:
            self.__source_dict = {file: source for file, source in self.__source_dict.items() if not self.__is_unittest(source)}

        self.__old_file = self.__open_old()
        if self.__old_file is None:
            self.__old_file = {self.__project_name: self.__source_dict, 'version': (self.__major, self.__minor, self.__build, self.__revision)}
        else:
            last_revision = self.__get_project_source()
            if last_revision is None:
                self.__old_file[self.__project_name] = self.__source_dict
                self.__increment_version()
            else:
                if last_revision != self.__source_dict:
                    self.__old_file[self.__project_name] = self.__source_dict
                    self.__increment_version()
                else:
                    if self.__enforce:
                        self.__increment_version()
                    else:
                        self.__major, self.__minor, self.__build, self.__revision = self.__get_version()

        self.__save_release()
        self.__save_as_new()

    def __get_python_files(self):
        extension_pattern = re.compile('.*\.pyc?', re.IGNORECASE)
        exclude_patterns = [re.compile(p, re.IGNORECASE) for p in EXCLUDE] if self.__exclude_patterns else [re.compile(p) for p in  TOTAL_EXCLUDE]
        res = []
        for path, _, files in os.walk(self.__project_path):
            for file in files:
                filepath = os.path.relpath(os.path.join(path, file))
                if not any([re.match(ep, filepath) for ep in exclude_patterns]):
                    if re.match(extension_pattern, filepath):
                        res.append(filepath)
        return res

    def __open_old(self):
        try:
            with open(os.path.join(self.__project_path, JSON_NAME), 'r') as last_file:
                return json.load(last_file)
        except (OSError, json.decoder.JSONDecodeError):
            return None

    def __get_project_source(self):
        try:
            return self.__old_file[self.__project_name]
        except KeyError:
            return None

    def __increment_version(self):
        curr_version = self.__get_version()
        if curr_version is None:
            return self.__major, self.__minor, self.__build, self.__revision
        else:
            pos = self.__release_types[self.__release_type]
            curr_version[pos] += 1
            for i in range(pos+1, 4):
                curr_version[i] = 0

            self.__major, self.__minor, self.__build, self.__revision = curr_version
            return self.__major, self.__minor, self.__build, self.__revision

    @staticmethod
    def __parse_source(filepath):
        with open(filepath, 'r') as file:
            source = file.read()
            return source

    @staticmethod
    def __is_unittest(source):
        source_lines = source.split('\n')
        test_patterns = [re.compile('.*import unittest.*'), re.compile(r'.*from unittest.*import.*')]
        for line in source_lines:
            if any(re.match(p, line) for p in test_patterns):
                return True
        return False

    def __save_as_new(self):
        with open(os.path.join(self.__project_path, JSON_NAME), 'w') as new_file:
            json.dump(self.__old_file, new_file, indent=2, sort_keys=True)

    def __get_version(self):
        try:
            return self.__old_file['version']
        except KeyError:
            return self.__major, self.__minor, self.__build, self.__revision

    def __save_release(self):
        release_path = os.path.join(self.__release_path, 'RELEASE_' + self.__project_name + '_{}_{}_{}_{}'.format(self.__major, self.__minor, self.__build, self.__revision))
        if os.path.exists(release_path):
            shutil.rmtree(release_path)
        shutil.copytree(self.__project_path, release_path, ignore=shutil.ignore_patterns('*RELEASE_' + self.__project_name + '_*'))
        self.__update_version(release_path)
        self.__add_notes(release_path)
        if self.__archive:
            shutil.make_archive(release_path, 'zip', release_path)
            shutil.rmtree(release_path)

    def __update_version(self, release_path):
        python_pattern = re.compile(r'.*\.py$')
        version_pattern_search = re.compile(VERSION_PATTERNS[0])
        version_pattern_sub = re.compile(VERSION_PATTERNS[1])

        for path, _, files in os.walk(release_path):
            for file in files:
                if re.match(python_pattern, file):
                    filepath = os.path.join(path, file)
                    file = self.__open_python_file(filepath)
                    if re.match(version_pattern_search, file):
                        version = "__version__ = '{}.{}.{}.{}'".format(self.__major, self.__minor, self.__build, self.__revision)
                        file = re.sub(version_pattern_sub, version, file)
                        self.__save_python_file(file, filepath)

        if len(self.__changed_files) > 0:
            print('Following files have been updated with current version ({}.{}.{}.{}):\n'.format(self.__major, self.__minor, self.__build, self.__revision))
            for file in self.__changed_files:
                print(file)

    def __add_notes(self, path):
        info = 'VERSION: {}.{}.{}.{}\n'.format(self.__major, self.__minor, self.__build, self.__revision)
        info += 'DATE: {}\n'.format(str(datetime.datetime.now()))
        if self.__message is not None:
            info += 'RELEASE MESSAGE:\n'
            info += str(self.__message)
        with open(os.path.join(path, '__RELEASE_INFO.txt'), 'w') as file:
            file.write(info)

    @staticmethod
    def __open_python_file(filepath):
        with open(filepath, 'r') as python_file:
            return python_file.read()

    def __save_python_file(self, file, filepath):
        self.__changed_files.append(filepath)
        with open(filepath, 'w') as python_file:
            python_file.write(file)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-r', '--release_type',  type=str, dest='release_type', help='release type [major/minor/build/revision]', default='build')
    parser.add_argument('-o', '--output', type=str, dest='output', help='output file path', default=None)
    parser.add_argument('--include_tests', dest='include_tests', help='tests are compared', action='store_false')
    parser.add_argument('--include_patterns', dest='include_patterns', help='special files are compared / not recommended /', action='store_false')
    parser.add_argument('-e', '--enforce', dest='enforce', help='enforces new version number', action='store_true')
    parser.add_argument('-a', '--archive', dest='archive', help='saves version as .zip archive', action='store_true')
    parser.add_argument('-m', '--message', dest='message', type=str, help='release comment', default=None)

    args = parser.parse_args()
    Versioner(release_type=args.release_type,
              release_path=args.output,
              exclude_tests=args.include_tests,
              exclude_patterns=args.include_patterns,
              enforce_new_version=args.enforce,
              archive=args.archive,
              message=args.message).version()
