# import subprocess
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
          metrics['ACL_FPR'] = float('.'.join(match[1]))
          metrics['ACL_ROC_Area'] = float('.'.join(match[6]))

      if 'nonACL' in line:
        # Extract FPR for nonACL class
        match = re.findall(r"(\d+).*?(\d+)", line)
        if match:
          metrics['nonACL_FPR'] = float('.'.join(match[1]))

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
  def __init__(self, java_path: str, weka_jar: str):
    if not self.__does_file_exist(java_path):
      raise FileNotFoundError(f'Java path not found: {java_path}')
    if not self.__does_file_exist(weka_jar):
      raise FileNotFoundError(f'Weka JAR file not found: {weka_jar}')
    self.java_path = java_path
    self.weka_jar = weka_jar
    self.arff_file = None
    self.classifiers = dict()
    self.rankers = dict()
    self.top_attributes = []

  def __does_file_exist(self, file_path: str) -> bool:
    print('Checking if file exists:', file_path)
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
    print('ARFF file:', self.arff_file)
    configs = self.__build_all_configs()
    print('\nNumber of configurations:', len(configs), end='\n\n')
  
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
    'ChiSquared': 'weka.attributeSelection.ChiSquaredAttributeEval'
  }

  top_attributes = [5, 6, 7, 8, 9, 10, 20, 50, 100, 200]

  load_dotenv()

  java_path = os.getenv('JAVA_PATH')
  weka_jar = os.getenv('WEKA_JAR_PATH')

  weka_runner = WekaController(java_path, weka_jar)
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

if __name__ == '__main__':
  main()