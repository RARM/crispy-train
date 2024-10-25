import subprocess
import re

def parse_weka_output(output_str):
  """Parses Weka's output to extract specific metrics.

  Args:
    output_str: The output string from Weka.

  Returns:
    A dictionary containing the extracted metrics.
  """

  metrics = {}
  output_str = output_str.split("=== Stratified cross-validation ===")
  output_str = output_str[-1].split("=== Confusion Matrix ===")[0]
  
  for line in output_str.splitlines():
    if " ACL" in line:
      # Extract FPR and ROC Area for ACL class
      match = re.findall(r"(\d+).*?(\d+)", line)
      if match:
        metrics['ACL_FPR'] = float('.'.join(match[1]))
        metrics['ACL_ROC_Area'] = float('.'.join(match[6]))

    if "nonACL" in line:
      # Extract FPR for nonACL class
      match = re.findall(r"(\d+).*?(\d+)", line)
      if match:
        metrics['nonACL_FPR'] = float('.'.join(match[1]))

  return metrics

# Assuming you have the Weka output in a string variable 'output_str'
# metrics = parse_weka_output(output_str)

# print("FPR for ACL:", metrics['ACL_FPR'])
# print("FPR for nonACL:", metrics['nonACL_FPR'])
# print("ROC Area for ACL:", metrics['ACL_ROC_Area'])

def get_macos_weka_call():
  app_dir = "/Applications/weka-3.9.6.app"
  java_path = app_dir + "/Contents/runtime/Contents/Home/bin/java"
  weka_jar = app_dir + "/Contents/app/weka.jar"
  return java_path, weka_jar

def run_weka_command(configuration, data_file_path) -> str:
  java_path, weka_jar = get_macos_weka_call()
  cmd = f"{java_path} -classpath {weka_jar} weka.classifiers.meta.AttributeSelectedClassifier -t {data_file_path} -x 10 -E \"weka.attributeSelection.GainRatioAttributeEval\" -S \"weka.attributeSelection.Ranker -T -1.0 -N 5\" -W weka.classifiers.bayes.NaiveBayes"
  print(cmd, end="\n\n")
  output = subprocess.run(cmd, shell=True, capture_output=True, text=True)
  return output

def main():
  configuration = "weka.classifiers.meta.AttributeSelectedClassifier -E \"weka.attributeSelection.GainRatioAttributeEval \" -S \"weka.attributeSelection.Ranker -T -1.0 -N 5\" -W weka.classifiers.bayes.NaiveBayes"
  data_file_path = "/Users/rodolfo/Projects/homework-aml/Lymphoma95x4023.arff"
  output = run_weka_command(configuration, data_file_path)
  # print(output.stderr)
  print(output.stdout)
  metrics = parse_weka_output(output.stdout)
  print("\n\n===== Extracted metrics:")
  print(metrics)
  print("FPR for ACL:", metrics['ACL_FPR'])
  print("FPR for nonACL:", metrics['nonACL_FPR'])
  print("ROC Area for ACL:", metrics['ACL_ROC_Area'])


if __name__ == "__main__":
  main()