from IPython.display import display, HTML, Javascript

class MetricsPlotter(pl.Callback):
    def __init__(self):
        super().__init__()
        self.epoch = 0
        self.metrics = {'train_loss': [np.inf], 'val_loss': [np.inf]}

    def set_template(self):
        css_style = f"""
          <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
          <style>
          .canvas-container {{
            display: flex;
            justify-content: center;
            width: 1000px;
          }}
          .chart-wrapper {{
            width: 800px;
            height: 500px;
          }}
          </style>

          <div class="canvas-container">
            <div class="chart-wrapper">
              <canvas id="chart0"></canvas>
            </div>
          </div>

          <script>
          const chartLabels = ['Train Loss', 'Validation Loss'];
          const chartOptions = {{'ticksStep': 1, 'yMax': 0.1}};

          const ctx = document.getElementById('chart0').getContext('2d');
          window.chart = new Chart(ctx, {{
              type: 'line',
              data: {{
                labels: Array.from({{length: {self.max_epochs}/100}}, (_, i) => (i + 1)*100),
                datasets: [{{
                  label: chartLabels[0],
                  data: [],
                  borderColor: 'blue',
                  backgroundColor: 'rgba(255, 0, 0, 0.1)',
                  borderWidth: 3,
                  pointStyle: false,
                }}, {{
                  label: chartLabels[1],
                  data: [],
                  borderColor: 'red',
                  backgroundColor: 'rgba(0, 0, 255, 0.1)',
                  borderWidth: 3,
                  pointStyle: false,
                }}]
              }},
              options: {{
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {{
                  y: {{
                    type: 'logarithmic',  // Set y-axis to logarithmic scale
                    min: 0.000001,  // Set a sensible minimum for log scale
                    max: chartOptions['yMax'],
                    ticks: {{
                      stepSize: chartOptions['ticksStep'],
                      callback: function(value, index, values) {{
                        if (value === 0.000001) return '1e-6';
                        return value;
                      }}
                    }},
                    grid: {{
                      color: '#aaaaaa',
                      lineWidth: 1,
                      drawBorder: true,
                      drawOnChartArea: true
                    }}
                  }},
                  x: {{
                    min: 0.0,
                    max : {self.max_epochs},
                    grid: {{
                      color: '#eeeeee',
                      lineWidth: 1,
                      drawBorder: true,
                      drawOnChartArea: true
                    }}
                  }}
                }}
              }}
            }});

          // update chart
          function uc(loss, val_loss) {{
            window.chart.data.datasets[0].data.push(loss);
            window.chart.data.datasets[1].data.push(val_loss);
            window.chart.update();
          }}
          </script>
          """
        display(HTML(css_style))

    def on_validation_epoch_end(self, trainer, pl_module):

        self.max_epochs = trainer.max_epochs
        # Assuming you log training loss on each batch
        train_loss = trainer.logged_metrics.get('train_loss', 0)
        if isinstance(train_loss, torch.Tensor):
            train_loss = train_loss.cpu().numpy().item()
        self.metrics['train_loss'].append(train_loss)

        val_loss = trainer.logged_metrics.get('val_loss', 0)
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.cpu().numpy().item()
        self.metrics['val_loss'].append(val_loss)

        self.plot_metrics(trainer)

    def plot_metrics(self, trainer):
        if self.epoch == 0:
            self.set_template()

        # Update the chart with new loss and val_loss values
        display(Javascript(f'''
        uc({self.metrics['train_loss'][-1]}, {self.metrics['val_loss'][-1]});
        '''))
        self.epoch = trainer.current_epoch
