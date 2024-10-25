# import subprocess
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
    attributes = []
    lines = output_str.splitlines()
    for i, line in enumerate(lines):
      if "Ranked attributes:" in line:
        for j in range(i + 1, i + num + 1):
          attributes.append(lines[j].split(' ')[-1])
        return attributes[:num]
    return attributes
