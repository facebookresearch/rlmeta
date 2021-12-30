Run PPO Algorithm Example for Atari Games
=========================================

Train PPO model with PongNoFrameskip-v4 environment for 20 epochs

.. code-block:: bash

   $ cd examples/atari/ppo
   $ python atari_ppo.py env="PongNoFrameskip-v4" num_epochs=20

The training logs and checkpoints will be saved to

.. code-block:: bash

   ./outputs/{YYYY-mm-dd}/{HH:MM:SS}/

We define the configs by `hydra <https://hydra.cc/>`_. The configs are defined
in

.. code-block:: bash

   ./conf/conf_ppo.yaml

After training, we can draw the training curve by run

.. code-block:: bash

   $ python ../../plot.py --log_file=./outputs/{YYYY-mm-dd}/{HH:MM:SS}/atari_ppo.log --fig_file=./atari_ppo.png --xkey=time

One example of the training curve is shown below.

.. image:: ./_static/img/atari_ppo.png
   :width: 600
