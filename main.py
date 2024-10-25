# import subprocess
import os
import re

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

class WekaRunner:
  def __init__(self, java_path: str, weke_jar: str):
    if not self.__does_file_exist(java_path):
      raise FileNotFoundError(f'Java path not found: {java_path}')
    if not self.__does_file_exist(weke_jar):
      raise FileNotFoundError(f'Weka JAR file not found: {weke_jar}')
    self.java_path = java_path
    self.weka_jar = weke_jar
    self.arff_file = None
    self.rankers = []
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
  
  def print_configuration(self):
    pass
  
def main():
  print('Hello world!')

if __name__ == '__main__':
  main()