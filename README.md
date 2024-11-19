
# Network Displayer

Display neural network output after learning


## Features

- Display neural network output after learning
- Help to see how NN learned


## Licence

[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)

## Usage/Examples

```python
nDisplayer = network_displayer.NetworkDisplayer(netClass=(-1, 1))
nDisplayer.draw_net_repr(net, "#2a9d8f", "#f4a261", nb_points=10000, bounds=(-0.5, -1.5, 1, 1.5))
nDisplayer.draw_inputs(inputs, targets, "#e76f51", "#264653")
nDisplayer.plot_show()
```


## Authors

- [Guillaume Seimandi](https://www.github.com/iSeaox)

