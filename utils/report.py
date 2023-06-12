"""

Report generation for classification

"""
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import os

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.python.keras.callbacks import Callback

REPORT_HTML_BASE = """
<!DOCTYPE html>
<html>
<head>
<title>Report</title>
<style>
table, th, td {{
  border: 1px solid;
}}
</style>
</head>
<body>
<h1>Report</h1>
{}
{}
{}
</body>
</html>
"""

IMAGE_HTML_BASE = """
<div class="imgBlock">
  <figure>
    <img src="{}" width="700" height="400"/>
  </figure>
</div>
"""


class Report:

    def __init__(self, output_dir, name, model):

        self.workdir = os.path.join(output_dir, name)
        self._init_folder()
        self.summary = self._generate_model_string(model)
        self.static_outputs = []

    def _init_folder(self):
        os.makedirs(self.workdir)
        os.mkdir(os.path.join(self.workdir, "static"))

    def save(self, start_time, step, epochs):
        header = "<p>Epochs: {}/{}</p>\n<p>Duration: {}</p>"\
            .format(step, epochs, str(timedelta(seconds=(time.time() - start_time))))
        image_htmls = list(map(lambda x: IMAGE_HTML_BASE.format(f"static/{x}"), self.static_outputs))
        summary_html = "\n".join(list(map(lambda x: f"<p>{x}</p>", self.summary.split("\n"))))
        html = REPORT_HTML_BASE.format(header, summary_html, "\n".join(image_htmls))
        with open(os.path.join(self.workdir, "report.html"), "w+") as f:
            f.write(html)

    @staticmethod
    def _generate_model_string(model):
        buffer = []
        model.summary(print_fn=lambda x: buffer.append(x))
        return "\n".join(buffer)

    def generate_metrics_plot(self, history):
        for metric in history.keys():
            plt.clf()
            plt.plot(history[metric])
            plt.xlabel("epochs")
            plt.ylabel(metric)
            plt.savefig(os.path.join(self.workdir, f"static/{metric}.png"))
            self.static_outputs.append(f"{metric}.png")

    def generate_correlation_matrix(self, y_true, y_pred, classes, one_hot=False):
        if one_hot:
            y_true = y_true.argmax(axis=1)
            y_pred = y_pred.argmax(axis=1)
        confusion_mat = confusion_matrix(y_true, y_pred)
        plt.clf()
        confusion_plt = ConfusionMatrixDisplay(confusion_mat, display_labels=classes)
        confusion_plt.plot()
        plt.savefig(os.path.join(self.workdir, f"static/confusion.png"))
        self.static_outputs.append("confusion.png")