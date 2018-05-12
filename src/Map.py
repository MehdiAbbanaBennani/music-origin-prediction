import plotly.graph_objs as go
from plotly.offline import plot
import json
from mapbox_token import mapbox_access_token
from math import pi
from parameters import PREDICT_FILE


class Map:
  def __init__(self, data_dir=PREDICT_FILE):
    with open(data_dir) as f:
      data = json.load(f)

    for k, v in data.items():
      setattr(self, k, v)

  @staticmethod
  def to_deg(rad):
    return rad / pi * 180

  def convert_list(self, preds):
    return [str(self.to_deg(pred)) for pred in preds]

  def transform_data(self, type):
    if type == "train" :
      correct_pred = ["Correct" if self.y_train_pred[i] == self.y_train[i] else "Wrong"
                      for i in range(len(self.y_train_pred))]
      latitudes = [pred[0] for pred in self.y_train_pred]
      longitudes = [pred[1] for pred in self.y_train_pred]
    if type == "test" :
      correct_pred = ["Correct" if self.y_test_pred[i] == self.y_test[i] else "Wrong"
                      for i in range(len(self.y_test_pred))]
      latitudes = [pred[0] for pred in self.y_test_pred]
      longitudes = [pred[1] for pred in self.y_test_pred]
    return self.convert_list(latitudes), self.convert_list(longitudes), correct_pred

  def process_data(self, type):
    latitudes, longitudes, correct_pred = self.transform_data(type=type)

    return [
      go.Scattermapbox(
          lat=latitudes,
          lon=longitudes,
          mode='markers',
          marker=dict(
              size=9
          ),
          text=correct_pred,
      )
    ]

  def generate(self, data, filename="map.html"):
    layout = go.Layout(
        autosize=True,
        hovermode='closest',
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=dict(
                lat=38.92,
                lon=-77.07
            ),
            pitch=0,
            zoom=4
        ),
    )

    fig = dict(data=data, layout=layout)
    plot(fig, filename=filename)

  def run(self, type):
    data = self.process_data(type)
    self.generate(data)