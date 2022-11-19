from typing import Callable, List, Optional, Type, Union
from torch import Tensor

import torch.nn as nn
from torch import flatten


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """_summary_

    Args:
        in_channels (int): _description_
        out_channels (int): _description_
        stride (int, optional): _description_. Defaults to 1.
        groups (int, optional): _description_. Defaults to 1.
        dilation (int, optional): _description_. Defaults to 1.

    Returns:
        nn.Conv2d: _description_
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """_summary_

    Args:
        in_channels (int): _description_
        out_channels (int): _description_
        stride (int, optional): _description_. Defaults to 1.

    Returns:
        nn.Conv2d: _description_
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class ResidualBlock(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_

    Raises:
        ValueError: _description_
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        down_sample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        """_summary_

        Args:
            in_channels (int): _description_
            out_channels (int): _description_
            stride (int, optional): _description_. Defaults to 1.
            down_sample (Optional[nn.Module], optional): _description_. Defaults to None.
            groups (int, optional): _description_. Defaults to 1.
            base_width (int, optional): _description_. Defaults to 64.
            dilation (int, optional): _description_. Defaults to 1.
            norm_layer (Optional[Callable[..., nn.Module]], optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
            NotImplementedError: _description_
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """_summary_

        Args:
            x (Tensor): _description_

        Returns:
            Tensor: _description_
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """

    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        down_sample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_planes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """_summary_

        Args:
            x (Tensor): _description_

        Returns:
            Tensor: _description_
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        block: Type[Union[ResidualBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_planes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck) and module.bn3.weight is not None:
                    nn.init.constant_(module.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(module, ResidualBlock) and module.bn2.weight is not None:
                    nn.init.constant_(module.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[ResidualBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        """_summary_

        Args:
            block (Type[Union[ResidualBlock, Bottleneck]]): _description_
            planes (int): _description_
            blocks (int): _description_
            stride (int, optional): _description_. Defaults to 1.
            dilate (bool, optional): _description_. Defaults to False.

        Returns:
            nn.Sequential: _description_
        """
        norm_layer = self._norm_layer
        down_sample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != planes * block.expansion:
            down_sample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.in_planes,
                planes,
                stride,
                down_sample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        ]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """_summary_

        Args:
            x (Tensor): _description_

        Returns:
            Tensor: _description_
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = flatten(x, 1)
        x = self.fc(x)

        return x


class ResNet50(ResNet):
    """_summary_

    Args:
        ResNet (_type_): _description_
    """

    def __init__(
        self,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super(ResNet50, self).__init__(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
        )


class ResNet101(ResNet):
    """_summary_

    Args:
        ResNet (_type_): _description_
    """

    def __init__(
        self,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super(ResNet101, self).__init__(
            block=Bottleneck,
            layers=[3, 4, 23, 3],
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
        )


class ResNet152(ResNet):
    """_summary_

    Args:
        ResNet (_type_): _description_
    """

    def __init__(
        self,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super(ResNet152, self).__init__(
            block=Bottleneck,
            layers=[3, 8, 36, 3],
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
        )
