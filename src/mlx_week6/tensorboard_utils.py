import torch
from importlib import reload
import mlx_week6.utils as utils

reload(utils)
from torch.utils.tensorboard import SummaryWriter


def log_model_parameters_to_tensorboard(model, writer, epoch):
    """
    Logs the histogram of model parameters to Tensorboard.

    Parameters:
    - model: The PyTorch model from which to log parameters.
    - writer: An instance of `torch.utils.tensorboard.SummaryWriter`.
    - epoch: The current epoch or iteration number, used as the step value in Tensorboard.
    """
    for name, param in model.named_parameters():
        # Use the parameter's name as the tag for the histogram
        tag = name.replace(".", "/")

        writer.add_histogram(tag, param.clone().cpu().data.numpy(), epoch)


def log_loss(loss, writer, epoch, loss_type="unspecified"):
    writer.add_scalar(f"Loss/{loss_type}", loss, epoch)


def log_probs_to_tensorboard(writer, epoch, probs, y):
    """
    Logs the probabilities of the model to Tensorboard.

    Parameters:
    - model: The PyTorch model from which to log parameters.
    - writer: An instance of `torch.utils.tensorboard.SummaryWriter`.
    - epoch: The current epoch or iteration number, used as the step value in Tensorboard.
    - x_batch: The input batch to the model.
    - y_batch: The target batch to the model.
    """
    with torch.no_grad():

        writer.add_histogram("softmax", probs.clone().cpu().data.numpy(), epoch)
        writer.add_histogram("y_sample_view", y.clone().cpu().data.numpy(), epoch)


def log_gradient_to_tensorboard(model, writer, epoch):
    """
    Logs the gradient of the parameters to Tensorboard.

    Parameters:
    - model: The PyTorch model from which to log parameters.
    - writer: An instance of `torch.utils.tensorboard.SummaryWriter`.
    - epoch: The current epoch or iteration number, used as the step value in Tensorboard.
    """
    # for name, param in tiny_story_model.named_parameters():
    #     if param.requires_grad and param.grad is not None:
    #         writer.add_scalar(
    #             f"grad_norm/{name}", param.grad.norm(), n_batches
    #         )
    for name, param in model.named_parameters():
        # Use the parameter's name as the tag for the histogram
        tag = name.replace(".", "/")
        if param.requires_grad and param.grad is not None:
            grad = param.grad.clone().norm()
            _is_bad, is_inf, is_large = utils.is_bad(grad)
            if _is_bad or is_inf or is_large:
                print(
                    f"Bad gradient detected in {name}, is_bad? {_is_bad}, is_inf{is_inf}, is_large?{is_large} "
                )
            writer.add_scalar(f"grad_norm/{tag}", grad, epoch)
        # writer.add_histogram(tag + "/grad", param.grad.clone().cpu().data.numpy(), epoch)


################# Hooks #################


def get_writer(config, run_id):
    run_id = run_id + "_".join(
        [f"{k}_{v}".replace(".", "_") for k, v in config.items()]
    )
    return SummaryWriter(log_dir=f"./logs/{run_id}")


def get_unique_run_name(config, run_id):
    run_id = run_id + "_".join(
        [f"{k}_{v}".replace(".", "_") for k, v in config.items()]
    )
    return run_id


class TinyStoryHooks:

    def __init__(self, config, model, run_id):

        self.writer = get_writer(config, run_id)
        self.step = 0
        was_set = False
        self.layer_shortnames = {"attn_softmax": "block.attn.softmax_layer"}
        self.layer_tags = {
            "attn_softmax": self.layer_shortnames["attn_softmax"].replace(".", "/")
        }
        self.quantile_vals = torch.tensor([0.05, 0.5, 0.95, 0.99])
        for name, module in model.named_modules():
            # print(name)
            if name == self.layer_shortnames["attn_softmax"]:
                was_set = True
                module.register_forward_hook(self.attn_sam_forward_hook)
        assert was_set is True

    @staticmethod
    def compute_quantiles_and_counts(data, quantiles):
        quantile_values = torch.quantile(data, quantiles)
        counts = torch.zeros_like(quantile_values)
        for i, q in enumerate(quantile_values):
            counts[i] = torch.sum(data <= q)
        return quantile_values, counts

    def attn_sam_forward_hook(self, module, input, output):

        if isinstance(input, tuple):
            input = input[0]
        assert isinstance(input, torch.Tensor), f"Input is weird type {type(input)}"
        data = input.view(-1).clone().cpu().detach()
        outputs = output.view(-1).clone().cpu().detach()
        finite_data = data[~torch.isinf(data)]
        assert (
            outputs.shape == data.shape
        ), f"Output shape {outputs.shape} != {data.shape}"
        finite_outputs = outputs[~torch.isinf(data)]
        # assert only upper triangular matrix is infinit
        _is_bad, is_inf, is_large = utils.is_bad(finite_data)
        if _is_bad or is_inf or is_large:
            print(
                f"Bad values detected in {self.layer_tags['attn_softmax']}, is_bad? {_is_bad}, is_inf? {is_inf}, is_large? {is_large} "
            )
        # if self.step % 200:

        q_vals, q_counts = TinyStoryHooks.compute_quantiles_and_counts(
            finite_data, self.quantile_vals
        )
        q_vals_op, q_counts_op = TinyStoryHooks.compute_quantiles_and_counts(
            finite_outputs, self.quantile_vals
        )
        # [0.05, 0.10, 0.5, 0.95, 0.99]
        for (q, qo), v in zip(
            zip(q_vals, q_vals_op), ["5%", "10%", "50%", "95%", "99%"]
        ):
            self.writer.add_scalar(
                self.layer_tags["attn_softmax"] + "_inputs" + f"/quantiles/{v}",
                q,
                self.step,
            )
            self.writer.add_scalar(
                self.layer_tags["attn_softmax"] + "_outputs" + f"/quantiles/{v}",
                qo,
                self.step,
            )
            # print(f"Input Quantiles: {torch.round(q_vals, decimals=3)}")
            # print(f"Output Quantiles: {torch.round(q_vals_op, decimals=3)}")

        self.writer.add_histogram(
            self.layer_tags["attn_softmax"], finite_data, self.step
        )
        self.writer.add_histogram(
            self.layer_tags["attn_softmax"] + "_output",
            outputs,
            self.step,
        )

        self.step += 1
