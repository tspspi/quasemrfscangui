# Graphical user interface for realtime view of QUASEM RF scans

|     |     |
| --- | --- |
| ![](https://raw.githubusercontent.com/tspspi/quasemrfscangui/refs/heads/master/doc/screen001.png) | ![](https://raw.githubusercontent.com/tspspi/quasemrfscangui/refs/heads/master/doc/screen002.png) |

This is a very simple interface for the QUASEM RF scan run type
to visualize in realtime. It has been designed to allow for interactive
viewing of short scans as well as display of the last retained RF
scan so one can perform realtime adjustments on the setup.

It requires a running ```quakesrctrl``` control system instance attached
to the MQTT broker to fetch information and to perform control operations

## Installation

```
pip install --upgrade quasemrfscangui-tspspi
```

## Configuration file

An optional configuration file can be put into ```~/.config/quasemrfscangui.cfg```.
This file supplies MQTT broker configuration and default spans

```
{
	"broker" : {
		"broker" : "198.51.100.1",
		"user" : "exampleuser",
		"port" : 1883,
		"password" : "putyourbrokersecrethere",
		"basetopic" : "quasem/experiment"
	}
}
```
