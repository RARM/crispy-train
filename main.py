import subprocess
import threading
import os
import re
from dotenv import load_dotenv

class Parser:
  @staticmethod
  def get_metrics(output_str: str) -> dict:
    """
    Parses the output string from Weka and extracts the desired metrics.

    Args:
      output_str: The output string from Weka.
    
    Returns:
      A dictionary containing the extracted metrics.
    """

    metrics = {}
    output_str = output_str.split('=== Stratified cross-validation ===')
    output_str = output_str[-1].split('=== Confusion Matrix ===')[0]
    
    for line in output_str.splitlines():
      if ' ACL' in line:
        # Extract FPR and ROC Area for ACL class
        match = re.findall(r"(\d+).*?(\d+)", line)
        if match:
          metrics['FPR_type1err'] = float('.'.join(match[1]))
          metrics['ROC_Area'] = float('.'.join(match[6]))

      if 'nonACL' in line:
        # Extract FPR for nonACL class
        match = re.findall(r"(\d+).*?(\d+)", line)
        if match:
          metrics['FNR_type2err'] = float('.'.join(match[1]))

    return metrics
  
  @staticmethod
  def get_ranked_attributes(output_str: str, num: int) -> list:
    """
    Parses the output string from Weka and extracts the ranked attributes.

    Args:
      output_str: The output string from Weka.
      num: The number of attributes to extract.

    Returns:
      A list containing the extracted attributes.
    """
    attributes = []
    lines = output_str.splitlines()
    for i, line in enumerate(lines):
      if "Ranked attributes:" in line:
        for j in range(i + 1, i + num + 1):
          attributes.append(lines[j].split(' ')[-1])
        return attributes[:num]
    return attributes

class WekaController:
  def __init__(self, java_path: str, weka_jar: str, extra_packages: list):
    if not self.__does_file_exist(java_path):
      raise FileNotFoundError(f'Java path not found: {java_path}')
    if not self.__does_file_exist(weka_jar):
      raise FileNotFoundError(f'Weka JAR file not found: {weka_jar}')
    for package in extra_packages:
      if not self.__does_file_exist(package):
        raise FileNotFoundError(f'Extra package not found: {package}')
    self.extra_packages = extra_packages
    self.java_path = java_path
    self.weka_jar = weka_jar
    self.arff_file = None
    self.classifiers = dict()
    self.rankers = dict()
    self.top_attributes = []

  def __does_file_exist(self, file_path: str) -> bool:
    return os.path.exists(file_path)
  
  def set_arff_file(self, arff_file: str):
    self.arff_file = arff_file

  def set_feature_selection_evaluators(self, evaluators: list, num_attributes: list):
    self.rankers = evaluators
    self.top_attributes = num_attributes

  def set_classifiers(self, classifiers: list):
    self.classifiers = classifiers

  def __build_all_configs(self) -> list:
    configs = []
    for classifier_name, classifier_cmd in self.classifiers.items():
      for ranker_name, ranker_cmd in self.rankers.items():
        for attr_num in self.top_attributes:
          config = {
          'classifier': {
            'name': classifier_name,
            'command': classifier_cmd
          },
          'ranker': {
            'name': ranker_name,
            'command': ranker_cmd
          },
          'attrNum': attr_num
          }
          configs.append(config)
    return configs
  
  def print_configuration(self):
    print("\nPrinting configuration...")
    print('Java path:', self.java_path)
    print('Weka JAR:', self.weka_jar)
    print('ARFF file:', self.arff_file)
    print('Extra packages:')
    for package in self.extra_packages:
      print(f'  - {package}')
    configs = self.__build_all_configs()
    print('\nNumber of configurations:', len(configs))

  def run_experiments(self):
    commands = self.__build_all_configs()
    mt_runner = MultiThreadedWekaRunner(
      commands,
      self.java_path,
      self.weka_jar,
      self.extra_packages,
      self.arff_file
    )
    return mt_runner.run()

  @staticmethod
  def print_result(result: dict):
    print(f"\n\nConfiguration ID: {result['id']}")
    print(f"Classifier: {result['classifier']}")
    print(f"Ranker: {result['ranker']}")
    print(f"Number of attributes: {result['attributesNum']}")
    print(f"Attributes: {result['attributes']}")
    print(f"Metrics: {result['metrics']}")

class MultiThreadedWekaRunner:
  def __init__(self, commands: list, java_path: str, weka_jar: str, extra_packages: str, data_file: str):
    self.commands = commands
    self.shared_results = []
    self.java_path = java_path
    self.weka_jar = weka_jar
    self.data_file = data_file
    self.extra_packages = extra_packages
  
  def run(self):
    threads = []
    packages_str = self.__get_packagas_str()
    for i, command in enumerate(self.commands):
      thread = threading.Thread(target=self.__run_command, args=(i, command, packages_str))
      threads.append(thread)
      thread.start()
    
    for thread in threads:
      thread.join()
    print("\rAll threads completed.                  \n")
    return self.shared_results
  
  def __get_packagas_str(self) -> str:
    all_packages = self.extra_packages + [self.weka_jar]
    return ':'.join(all_packages)
  
  def __run_command(self, id: int, command: dict, packages_str: str):
    cmd = f"{self.java_path} -classpath {packages_str} weka.classifiers.meta.AttributeSelectedClassifier -t {self.data_file} -x 10 -E \"{command['ranker']['command']}\" -S \"weka.attributeSelection.Ranker -T -1.0 -N {command['attrNum']}\" -W {command['classifier']['command']}"
    output = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    metrics = Parser.get_metrics(output.stdout)
    attributes = Parser.get_ranked_attributes(output.stdout, command['attrNum'])
    self.shared_results.append({
      'id': id,
      'classifier': command['classifier']['name'],
      'ranker': command['ranker']['name'],
      'attributesNum': command['attrNum'],
      'attributes': attributes,
      'metrics': metrics,
    })
    print(f"\rCompleted {len(self.shared_results)} out of {len(self.commands)}.", end='')

class CSVWriter:
  @staticmethod
  def convert_weka_output_to_2d_list(output: dict) -> list:
    data = []
  
    heading = [
      'ID',
      'Classifier',
      'Ranker',
      'AttributesNum',
      'Attributes',
      'FPR_type1err',
      'FNR_type2err',
      'ROC_Area'
    ]

    data.append(heading)
    
    for result in output:
      row = [
        str(result['id']),
        result['classifier'],
        result['ranker'],
        str(result['attributesNum']),
        ';'.join(result['attributes']),
        str(result['metrics'].get('FPR_type1err',)),
        str(result['metrics'].get('FNR_type2err',)),
        str(result['metrics'].get('ROC_Area',))
      ]
      data.append(row)
    return data

  @staticmethod
  def save_results(data: list, output_file: str):
    data_list = CSVWriter.convert_weka_output_to_2d_list(data)
    with open(output_file, 'w') as file:
      for row in data_list:
        file.write(','.join(row) + '\n')
  
def main():
  classifiers_cmds = {
    'NaiveBayes': 'weka.classifiers.bayes.NaiveBayes',
    '5NN': "weka.classifiers.lazy.IBk -- -K 5 -W 0 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -R first-last\\\"\""
  }

  rankers = {
    'GainRatio': 'weka.attributeSelection.GainRatioAttributeEval',
    'SymmetricUncertainty': 'weka.attributeSelection.SymmetricalUncertAttributeEval',
    'ReliefF': 'weka.attributeSelection.ReliefFAttributeEval -M -1 -D 1 -K 10',
    'ReliefF-W': 'weka.attributeSelection.ReliefFAttributeEval -W -M -1 -D 1 -K 10 -A 2',
    'InformationGain': 'weka.attributeSelection.InfoGainAttributeEval',
    'ChiSquared': 'weka.attributeSelection.ChiSquaredAttributeEval '
  }

  top_attributes = [5, 6, 7, 8, 9, 10, 20, 50, 100, 200]

  load_dotenv()

  java_path = os.getenv('JAVA_PATH')
  weka_jar = os.getenv('WEKA_JAR_PATH')
  extra_packages = [
    os.getenv('CHI_PACKAGE_PATH')
  ]
  
  output_file = os.path.abspath('./output.csv')

  weka_runner = WekaController(java_path, weka_jar, extra_packages)
  weka_runner.set_arff_file(
    os.path.abspath('./Lymphoma95x4023.arff')
  )
  weka_runner.set_feature_selection_evaluators(
    rankers,
    top_attributes
  )
  weka_runner.set_classifiers(
    classifiers_cmds
  )

  weka_runner.print_configuration()
  results = weka_runner.run_experiments()
  CSVWriter.save_results(results, output_file)

if __name__ == '__main__':
  main()