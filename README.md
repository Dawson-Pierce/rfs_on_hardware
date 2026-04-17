# rfs_on_hardware

Run brew's RFS filters on hardware, fed simulated detections from a host PC over USB serial for initial tests.

# Getting started: Example for PHD filter on teensy

Clone brew:
```bash
git clone https://github.com/Dawson-Pierce/brew.git
```

Build protocol:
```bash
cd protocol/
python regen.py
```

Flash firmware and run test:
```bash
cd firmware/
python -m platformio run -t upload 
python ../host/sim_runner.py --port XXX
```

Note that XXX should be replaced by the port connected to the teensy. This can be found with the command:
```bash
python -m platformio device list
```

Variations of the tests can be operated with flags found in `host/sim_runner.py`. 
