res = []

inp = [
  {
    "total": {
      "target_stats": {
        "ll_p": 3.654968396,
        "auc": 0.9797376477,
        "ctr": 0.01005937883,
        "q_auc": 0.971656774,
        "ctr_factor": 1.000478909,
        "log_loss": 0.01950749777,
        "clicks": 4562326.369,
        "shows": 4.535395719E8
      }
    },
    "slices": [
      {
        "target_stats": {
          "ll_p": 3.363760825,
          "auc": 0.9654578561,
          "ctr": 0.006879490226,
          "q_auc": 0.9676134228,
          "ctr_factor": 1.014049264,
          "log_loss": 0.01796924388,
          "clicks": 916145.4767,
          "shows": 133170547
        },
        "slice_key": [
          "1d",
        ],
        "slice_value": [
          "1"
        ]
      },
      {
        "target_stats": {
          "ll_p": 2.780897593,
          "auc": 0.9645406642,
          "ctr": 0.0145836411,
          "q_auc": 0.9418011369,
          "ctr_factor": 1.006683672,
          "log_loss": 0.03557868439,
          "clicks": 1365017.708,
          "shows": 9.359923896E7
        },
        "slice_key": [
          "1d",
        ],
        "slice_value": [
          "0"
        ]
      },
    ]
  },
  {
    "total": {
      "target_stats": {
        "ll_p": 3.770124607,
        "auc": 0.9785523232,
        "ctr": 0.009534162545,
        "q_auc": 0.9555704366,
        "ctr_factor": 0.9674428767,
        "log_loss": 0.01790484227,
        "clicks": 4324120,
        "shows": 453539572
      }
    },
    "slices": [
      {
        "target_stats": {
          "ll_p": 2.960358768,
          "auc": 0.9560920787,
          "ctr": 0.0109348343,
          "q_auc": 0.9411826023,
          "ctr_factor": 1.101947833,
          "log_loss": 0.02788334147,
          "clicks": 1221823,
          "shows": 111736764
        },
        "slice_key": [
          "2d",
        ],
        "slice_value": [
          "1",
        ]
      },
      {
        "target_stats": {
          "ll_p": 3.222512121,
          "auc": 0.9714586435,
          "ctr": 0.008173626874,
          "q_auc": 0.8932741518,
          "ctr_factor": 0.8627507259,
          "log_loss": 0.02108985712,
          "clicks": 940237,
          "shows": 115033022
        },
        "slice_key": [
          "2d",
        ],
        "slice_value": [
          "0",
        ]
      },
    ]
  }
]

for i in range(len(inp)):
  slices = inp[i]["slices"]

  for slice_dict in slices:
    slice_metrics = slice_dict["target_stats"]
    slice_name = ", ".join(
      [name + "=" + value for name, value in zip(slice_dict["slice_key"], slice_dict["slice_value"])]
    )

    for metric_name, metric_value in slice_metrics.items():
      res.append({
        "name": metric_name,
        "slice": slice_name,
        "value": metric_value
      })

print(res)