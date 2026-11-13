"""e01: silence the dedicated 'wait' neuron L16.2431 (zero its down_proj column: 2048 weights)."""
def edit(model):
    model.model.layers[16].mlp.down_proj.weight[:, 2431] = 0
