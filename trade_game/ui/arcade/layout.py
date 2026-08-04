"""图形界面在不同窗口尺寸下的稳定布局计算。"""

from __future__ import annotations

from dataclasses import dataclass

from .theme import Rect


@dataclass(frozen=True, slots=True)
class MainLayout:
    """经营主屏的稳定工作区矩形。"""

    title: Rect
    status: Rect
    navigation: Rect
    map: Rect
    detail: Rect
    footer: Rect


def main_layout(width: int, height: int, *, route_view: bool = True) -> MainLayout:
    """按当前页面分配导航、路线图和主工作区。

    采购和出售需要完整宽度来展示订单信息，路线页才同时显示地图和行程面板。
    """

    margin = 14
    title_height = 34
    status_height = 74
    footer_height = 44
    navigation_width = max(144, min(176, int(width * 0.16)))
    title = Rect(margin, height - margin - title_height, width - margin, height - margin)
    status = Rect(margin, title.bottom - status_height - 8, width - margin, title.bottom - 8)
    footer = Rect(margin, margin, width - margin, margin + footer_height)
    content_bottom = footer.top + 10
    content_top = status.bottom - 10
    navigation = Rect(margin, content_bottom, margin + navigation_width, content_top)
    workspace = Rect(navigation.right + 10, content_bottom, width - margin, content_top)
    if route_view:
        map_width = max(420, int(workspace.width * 0.62))
        map = Rect(workspace.left, workspace.bottom, workspace.left + map_width, workspace.top)
        detail = Rect(map.right + 10, workspace.bottom, workspace.right, workspace.top)
    else:
        # 非路线页面不占用地图，避免把交易表格压进狭窄的侧栏。
        map = Rect(workspace.left, workspace.bottom, workspace.left, workspace.top)
        detail = workspace
    return MainLayout(
        title=title,
        status=status,
        navigation=navigation,
        map=map,
        detail=detail,
        footer=footer,
    )
