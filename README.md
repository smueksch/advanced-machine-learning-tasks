> This project was part of the [Advanced Machine Learning](https://ml2.inf.ethz.ch/courses/aml/) course in Fall 2021 at ETH Zürich.

# Setup

The following environment variable needs to be set in order to run any scripts
involving Comet.ml:

```
export COMET_API_KEY="<API key>"
```

The API key can be found on the Comet.ml project page.

# Running on Euler Cluster

In order to run an experiment on the Euler cluster, first execute the following
command to change to the new software stack:

```
env2lmod
```

Then, set the Comet.ml API key as described in _Setup_ above:

```
export COMET_API_KEY="<API key>"
```

After that, use one of the `run_*.sh` scripts to start the desired experiment or
manually configure the experiment execution.
