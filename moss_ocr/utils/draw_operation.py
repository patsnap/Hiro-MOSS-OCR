"""
@author: wwjiang
"""

import os
import sys
import cv2
import random
import numpy as np
import PIL
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt

__all__ = [
    "DrawImage",
    "random_color_bar",
    "COLOR_BAR",
    "draw_2d_line",
    "draw_bin_image",
]

COLOR_BAR = [
    "#0000FF", "#FF0000", "#008000", "#008080", "#800080", "#808000",
    "#6495ED", "#DE3163", "#FF7F50", "#CCCCFF", r"#FFBF00", "#40E0D0", "#DFFF00", "#9FE2BF",
    '#0000FF', '#04B431', '#FF7F50', '#FF3838', '#FF6347', '#CB38FF', '#0018EC', '#C76114', '#FF9D97', '#03A89E',
    '#FF701F', '#FFB21D', '#00C78C', '#B0171F', '#CFD231', '#48F90A', '#92CC17', '#3DDB86', '#1A9334', '#00D4BB',
    '#2C99A8', '#00C2FF', '#344593', '#6473FF', '#F0FFFF', '#FF37C7', '#8438FF', '#520085', '#FF95C8'
]

curdir = os.path.dirname(__file__)
rtpath = os.path.join(curdir, "../..")
sys.path.append(rtpath)

FONT_PATH = os.path.join(rtpath, "ocr_anything/static/fonts/Deng.ttf")


class DrawImage:

    def __init__(self):
        pass

    def polt(self, x, y_ls, x_label, y_label_ls, title, default_color=COLOR_BAR[0], circulate_color=True):
        raise NotImplementedError

    def rectangles(
            self,
            img: np.ndarray,
            boxes: list | tuple | np.ndarray,
            thickness: int = 2,
            default_color: tuple | str = COLOR_BAR[0],
            circulate_color: bool = True
    ) -> np.ndarray:
        """
        Args:
            img:
            boxes: [[x1, y1, x2, y2], ...] or [[x1, y1, x2, y2, x3, y3, x4, y4], ...],
            thickness:
            default_color:
            circulate_color:
        """
        def check_boxes(_boxes: list | tuple | np.ndarray) -> list | tuple:
            if isinstance(_boxes, np.ndarray):
                _boxes = _boxes.tolist()
            if not isinstance(_boxes[0], (list, tuple)):
                _boxes = [_boxes]
            if len(_boxes[0]) == 8:
                boxes_new = []
                for _box in _boxes:
                    x1, y1, x2, y2, x3, y3, x4, y4 = _box[:8]
                    boxes_new.append([x1, y1, x3, y3])
                return boxes_new
            assert len(_boxes[0]) == 4
            return _boxes

        if isinstance(default_color, str):
            default_color = self.hex2rgb(default_color)
        img = img.copy()
        img = self.check_img(img)
        boxes = check_boxes(boxes)

        for idx, box in enumerate(boxes):
            if circulate_color:
                cur_color = self.hex2rgb(COLOR_BAR[idx % len(COLOR_BAR)])
            else:
                cur_color = default_color
            img = cv2.rectangle(img, tuple(box[:2]), tuple(box[2:]), color=cur_color, thickness=thickness)
        return img

    def polygon(
            self,
            img: np.ndarray,
            boxes: list | tuple | np.ndarray,
            thickness: int = 2,
            default_color: str | tuple = COLOR_BAR[0],
            circulate_color: bool = True
    ) -> np.ndarray:
        """
        Args:
            img:
            boxes: [[[x1, y1], [x2, y2], ...], [[x1, y1], [x2, y2], ...]]
            thickness:
            default_color:
            circulate_color:
        """

        def check_boxes(_boxes: list | tuple | np.ndarray) -> list | tuple:
            if isinstance(_boxes, np.ndarray):
                _boxes = _boxes.tolist()
            if not isinstance(_boxes[0][0], (list, tuple)):
                _boxes = [_boxes]
            return _boxes

        if isinstance(default_color, str):
            default_color = self.hex2rgb(default_color)
        img = img.copy()
        img = self.check_img(img)
        boxes = check_boxes(boxes)

        for idx, box in enumerate(boxes):
            if circulate_color:
                cur_color = self.hex2rgb(COLOR_BAR[idx % len(COLOR_BAR)])
            else:
                cur_color = default_color
            cur_box = np.array(box).reshape(-1, 1, 2).astype(np.int32)
            img = cv2.polylines(img, [cur_box], True, color=cur_color, thickness=thickness)
        return img

    def circles(
            self,
            img: np.ndarray, pts: list | tuple | np.ndarray,
            radius: int = 6,
            thickness: int = 2,
            default_color: str | tuple[int, int, int] = COLOR_BAR[0],
            circulate_color: bool = True
    ) -> np.ndarray:
        """
        draw some circles in the images.
        Args:
            pts[list | tuple | np.ndarray]: center of circle. [(x1, y1), (x2, y2)...]
        """
        if isinstance(default_color, str):
            default_color = self.hex2rgb(default_color)
        img = img.copy()
        img = self.check_img(img)
        if isinstance(pts, np.ndarray):
            pts = pts.tolist()
        if not isinstance(pts[0], (list, tuple)):  # multiple circles
            pts = [pts]
        for idx, pt in enumerate(pts):
            if circulate_color:
                cur_color = self.hex2rgb(COLOR_BAR[idx % len(COLOR_BAR)])
            else:
                cur_color = default_color
            img = cv2.circle(img=img, center=pt, radius=radius, thickness=thickness, color=cur_color)
        return img

    def line(self, img, points: list[list] | np.ndarray, thickness=2, default_color=(0, 0, 255),
             circulate_color=False, with_circle=True, solid_circle=True):
        """
        points:
            if np.ndarray, dim: N * 4 (start_x, start_y, end_x, end_y)
            if list[list[float | int]], each item [tart_x, start_y, end_x, end_y]
        """

        def check_points(_points):
            if isinstance(_points, list):
                _points = np.array(_points).reshape(-1, 4)
            elif isinstance(_points, np.ndarray):
                assert _points.shape[1] == 4 and _points.ndim == 2, f"but received: {_points.shape}"
            else:
                raise TypeError(f"points only support list or ndarray, but received {type(_points)}")
            return points.tolist()

        img = self.check_img(img)
        img = img.copy()
        points_ls = check_points(_points=points)
        for idx, point in enumerate(points_ls):
            assert point.__len__() == 4, f"point only have for coordinate, but received {point}"
            x1, y1, x2, y2 = point[:4]
            if circulate_color:
                cur_color = self.hex2rgb(COLOR_BAR[idx % len(COLOR_BAR)])
            else:
                cur_color = default_color
            img = cv2.line(img, (x1, y1), (x2, y2), color=cur_color, thickness=thickness)

            if with_circle:
                circle_color = (np.array(cur_color) * 0.8 + np.array((0, 0, 0)) * 0.2).astype(np.uint8).tolist()
                img = self.circles(img, [x1, y1], radius=int(thickness * 2),
                                   thickness=-1 if solid_circle else thickness, default_color=circle_color)
                img = self.circles(img, [x2, y2], radius=int(thickness * 2),
                                   thickness=-1 if solid_circle else thickness, default_color=circle_color)
        return img

    def rectangle_with_annotation_new(
            self,
            img: np.ndarray,
            boxes: list[list],
            labels: list[str],
            font_path: str | None = None,
            font_size: int = 16,
            thickness: int = 2,
    ) -> np.ndarray:
        """draw rectangle with annotations
            Args:
                img(ndarray)
                boxes(list[list]): [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...]
                labels(list[str]): [str1, str2,...],
                font_path:
                font_size:
                thickness:
            Return
        """
        assert isinstance(boxes, (list, tuple))
        assert isinstance(boxes[0], (list, tuple))
        assert isinstance(labels, (list, tuple))
        assert len(boxes) == len(labels)
        img = img.copy()
        img = self.check_img(img)
        anno_info = []

        for idx, (box, label) in enumerate(zip(boxes, labels)):
            cur_color = self.hex2rgb(COLOR_BAR[idx % len(COLOR_BAR)])
            img = cv2.rectangle(img, tuple(box[:2]), tuple(box[2:]), color=cur_color, thickness=thickness)
            anno_info.append([box[0], box[1], label, cur_color, self.rect_vertical_classifier(box)])

        if font_path is None:
            cur_font = ImageFont.truetype(FONT_PATH, font_size)
        else:
            cur_font = ImageFont.truetype(font_path, font_size)

        img_pil = Image.fromarray(img)
        for idx, (lb_x, lb_y, label, color, is_vertical) in enumerate(anno_info):
            self.__draw_text(img_pil, int(is_vertical), label, [lb_x, lb_y], cur_font, color=color)
        return np.array(img_pil)

    @classmethod
    def generate_color_seq(cls, _labels: list):
        color_seq = []
        label_color_map = {}
        label_count = 0
        for _idx, _cur_label in enumerate(_labels):
            if _cur_label not in label_color_map:
                _cur_color = cls.hex2rgb(COLOR_BAR[label_count % len(COLOR_BAR)])
                color_seq.append(_cur_color)
                label_color_map[_cur_label] = _cur_color
                label_count += 1
            else:
                color_seq.append(label_color_map[_cur_label])
        return color_seq

    @classmethod
    def poly_with_annotation_new(
            cls,
            img: np.ndarray,
            boxes: list | tuple | np.ndarray,
            labels: list | tuple | np.ndarray,
            font_path: str | None = None,
            font_size: int = 16,
            thickness: int = 2,
            color: list | None = None,
            font_color: tuple | None = None,
            draw_vertical_text: bool = True,
            **kwargs
    ) -> np.ndarray:
        """draw rectangle with annotations
        boxes:
            when List: [_DIM(4, 2), _DIM(4, 2), ...] ; _DIM(4, 2) info: [(x1, y1), (x2, y2), ...]
            when ndarray: _DIM(N, 4, 2)
        """

        def check_boxes(_boxes: list | tuple | np.ndarray) -> list | tuple:
            if isinstance(_boxes, np.ndarray):
                _boxes = _boxes.tolist()
            assert isinstance(_boxes, list | tuple)
            return _boxes

        def check_labels(_labels: list | tuple | np.ndarray) -> list | tuple:
            assert isinstance(_labels, list | tuple | np.ndarray)
            if isinstance(_labels, np.ndarray):
                _labels = _labels.tolist()
            return _labels

        img = img.copy()
        img = cls.check_img(img)
        boxes = check_boxes(boxes)
        labels = check_labels(labels)
        if color is None:
            color_seq = cls.generate_color_seq(labels)
        else:
            color_seq = color
        anno_info = []

        for idx, (box, label, cur_color) in enumerate(zip(boxes, labels, color_seq)):
            cur_box = np.array(box).reshape(-1, 1, 2).astype(np.int32)
            img = cv2.polylines(img, [cur_box], True, color=cur_color, thickness=thickness)
            anno_info.append(
                [
                    cur_box[0][0][0],
                    cur_box[0][0][1],
                    label,
                    cur_color,
                    cls.poly_vertical_classifier(cur_box) if draw_vertical_text else False
                ])

        if font_path is None:
            cur_font = ImageFont.truetype(FONT_PATH, font_size)
        else:
            cur_font = ImageFont.truetype(font_path, font_size)

        img_pil = Image.fromarray(img)
        for lb_x, lb_y, label, color, is_vertical in anno_info:
            cls.__draw_text(
                img_pil,
                direction=int(is_vertical),
                text=label,
                start_pos=[lb_x, lb_y],
                font=cur_font,
                color=color,
                font_color=font_color
            )
        return np.array(img_pil)

    @classmethod
    def poly_with_annotation_contrastive_show(
            cls,
            img: np.ndarray,
            boxes: list | tuple | np.ndarray,
            labels: list | tuple | np.ndarray,
            font_path: str | None = None,
            color: list | None = None,
            font_color: tuple | None = None,
            font_size: int | None = None,
            thickness: int = 2,
    ) -> np.ndarray:
        """draw rectangle with annotations
        boxes:
            when List: [_DIM(4, 2), _DIM(4, 2), ...] ; _DIM(4, 2) info: [(x1, y1), (x2, y2), ...]
            when ndarray: _DIM(N, 4, 2)
        """

        def check_boxes(_boxes: list | tuple | np.ndarray) -> list | tuple:
            if isinstance(_boxes, np.ndarray):
                _boxes = _boxes.tolist()
            assert isinstance(_boxes, (list, tuple))
            return _boxes

        def check_labels(_labels: list | tuple | np.ndarray) -> list | tuple:
            assert isinstance(_labels, (list, tuple, np.ndarray))
            if isinstance(_labels, np.ndarray):
                _labels = _labels.tolist()

            if isinstance(_labels, (list, tuple)):
                _labels = [f"{i:.2f}" if isinstance(i, (float, int)) else i for i in _labels]
            return _labels

        img = img.copy()
        img = cls.check_img(img)
        img_draw = np.ones_like(img, dtype=np.uint8) * 255
        boxes = check_boxes(boxes)
        labels = check_labels(labels)
        if color is None:
            color_seq = cls.generate_color_seq(labels)
        else:
            color_seq = color
        anno_info = []

        for idx, (box, label, cur_color) in enumerate(zip(boxes, labels, color_seq)):
            cur_box = np.array(box).reshape(-1, 1, 2).astype(np.int32)
            _, (box_w, box_h), _ = cv2.minAreaRect(cur_box)
            img_draw = cv2.polylines(img_draw, [cur_box], True, color=cur_color, thickness=thickness)
            anno_info.append(
                [cur_box[0][0][0],
                 cur_box[0][0][1],
                 label,
                 cur_color,
                 cls.poly_vertical_classifier(cur_box),
                 (box_w, box_h)
                 ]
            )

        img_pil = Image.fromarray(img_draw)
        for lb_x, lb_y, cur_label, color, is_vertical,  box_wh in anno_info:
            cls.draw_inner_box_text(
                img_pil,
                text=cur_label,
                direction=int(is_vertical),
                start_pos=[lb_x, lb_y],
                box_wh=box_wh,
                color=color,
                font_color=font_color,
                font_path=font_path,
                font_size=font_size
            )
        img_draw = np.array(img_pil)
        return np.concatenate([img, img_draw], axis=1)

    @classmethod
    def draw_inner_box_text(
            cls,
            img_pil: Image.Image,
            text: str,
            direction: int | None = None,
            start_pos: list | tuple | None = None,
            box_wh: list | tuple | None = None,
            draw_outline: bool = True,
            font_path: str | None = None,
            color: tuple[int, int, int] = (0, 0, 255),
            font_color: tuple | None = None,
            font_size: int | None = None
    ):
        """ draw text in canvas
        direction(int):
            0: draw horizontal text
            1: draw vertical text
        """
        if box_wh is None:
            box_wh = (img_pil.width, img_pil.height)
        if start_pos is None:
            start_pos = [1, 1]
        if direction is None:
            w, h = box_wh
            direction = cls.__vertical_classifier(w/h)
        font_size = font_size if font_size is not None else max(int(min(box_wh) * 0.65), 11)

        if font_path is None:
            font = ImageFont.truetype(FONT_PATH, font_size)
        else:
            font = ImageFont.truetype(font_path, font_size)

        draw = ImageDraw.Draw(img_pil)

        img_size = img_pil.size
        if PIL.__version__.split(".")[0] == '9':
            char_wh_info = [font.getsize(i)[:2] for i in text]
        else: # PIL.__version__.split(".")[0] == '10':
            char_wh_info = [font.getbbox(i)[-2:] for i in text]

        start_pos[1 - direction] = max(start_pos[1 - direction], 0)

        if direction:
            rect = [
                max(start_pos[0] - 1, 0),
                start_pos[1],
                min(start_pos[0], img_size[0]),
                min(start_pos[1] + sum([i[1] for i in char_wh_info]), img_size[1])
            ]
        else:
            rect = [
                start_pos[0],
                max(start_pos[1] - 1, 0),
                min(start_pos[0] + sum([i[0] for i in char_wh_info]), img_size[0]),
                min(start_pos[1], img_size[1])
            ]
        rect_color = (np.array([255, 255, 255]) * 0.4 + np.array(color) * 0.6).astype(np.uint8)
        if draw_outline:
            draw.rectangle(rect, fill=tuple(rect_color.tolist()))
        for idx, cur_char in enumerate(text):
            draw.text(
                start_pos,
                cur_char,
                fill=tuple([255 - i if font_color is None else i for i in (color if font_color is None else font_color)]),
                font=font
            )
            start_pos[direction] += char_wh_info[idx][direction]
            if start_pos[direction] > img_size[direction]:
                start_pos[direction] = img_size[direction]
                break
        rect.extend([start_pos[0], start_pos[1]])

    @classmethod
    def draw_vertical_text(cls, img_pil: Image.Image, text: str, start_pos: list | tuple, font: ImageFont,
                           draw_outline=True, color=(0, 0, 255), font_color=None):
        cls.__draw_text(img_pil, direction=1, text=text, start_pos=start_pos, font=font, draw_outline=draw_outline,
                        color=color, font_color=font_color)

    @classmethod
    def draw_horizonal_text(
            cls,
            img_pil: Image.Image,
            text: str, start_pos: list | tuple,
            font: ImageFont,
            draw_outline=True,
            color=(0, 0, 255),
            font_color=None
    ):
        cls.__draw_text(img_pil, direction=0, text=text, start_pos=start_pos, font=font, draw_outline=draw_outline,
                        color=color, font_color=font_color)

    @staticmethod
    def __draw_text(
            img_pil: Image.Image,
            direction: int,
            text: str,
            start_pos: list,
            font: ImageFont,
            draw_outline=True,
            color=(0, 0, 255),
            font_color: tuple[int, int, int] | None = None
    ):
        """ draw text in canvas
        direction(int):
            0: draw horizontal text
            1: draw vertical text
        """
        if len(text) <= 0:
            return
        draw = ImageDraw.Draw(img_pil)
        img_size = img_pil.size
        if PIL.__version__.split(".")[0] == '9':
            char_wh_info = [font.getsize(i)[:2] for i in text]
        else:  # PIL.__version__.split(".")[0] in ('10', '11'):
            char_wh_info = [font.getbbox(i)[-2:] for i in text]
        offset = max(char_wh_info, key=lambda x: x[1 - direction])[1 - direction]
        start_pos[1 - direction] = max(start_pos[1 - direction] - offset, 0)

        if direction:
            rect = [
                max(start_pos[0] - 1, 0),
                start_pos[1],
                min(offset + start_pos[0], img_size[0]),
                min(start_pos[1] + sum([i[1] for i in char_wh_info]), img_size[1])
            ]
        else:
            rect = [
                start_pos[0],
                max(start_pos[1] - 1, 0),
                min(start_pos[0] + sum([i[0] for i in char_wh_info]), img_size[0]),
                min(offset + start_pos[1], img_size[1])
            ]
        rect_color = (np.array([255, 255, 255]) * 0.4 + np.array(color) * 0.6).astype(np.uint8)
        if draw_outline:
            draw.rectangle(rect, fill=tuple(rect_color.tolist()))
        for idx, cur_char in enumerate(text):
            draw.text(
                start_pos,
                cur_char,
                fill=tuple([255 - i if font_color is None else i for i in (color if font_color is None else font_color)]),
                font=font
            )
            start_pos[direction] += char_wh_info[idx][direction]
            if start_pos[direction] > img_size[direction]:
                start_pos[direction] = img_size[direction]
                break
        rect.extend([start_pos[0], start_pos[1]])

    @classmethod
    def rect_vertical_classifier(cls, bbox: list, tolerate=2.5) -> bool:
        x1, y1, x2, y2 = bbox[:4]
        h = abs(y2 - y1)
        w = abs(x2 - x1)
        return cls.__vertical_classifier(h / w, tolerate=tolerate)

    @classmethod
    def poly_vertical_classifier(cls, bbox: np.ndarray, tolerate: float = 2.5) -> bool:
        """bbox shape like (N, 1, 2)"""
        bbox = bbox.squeeze(axis=1)
        xmin, xmax = min(bbox[:, 0]), max(bbox[:, 0])
        ymin, ymax = min(bbox[:, 1]), max(bbox[:, 1])
        h, w = abs(ymax - ymin), abs(xmax - xmin)
        return cls.__vertical_classifier(h / w, tolerate=tolerate)

    @staticmethod
    def __vertical_classifier(hw_ratio, tolerate: float = 2.5):
        if hw_ratio > tolerate:
            return True
        else:
            return False

    @staticmethod
    def check_img(img: np.ndarray | Image.Image) -> np.ndarray:
        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img, np.ndarray)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=2)
        assert img.ndim == 3
        return img

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def hex2bgr(h):
        h = h[1:]  # remove #
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (4, 2, 0))


def random_color_bar(seed=0):
    random.seed(seed)
    shuffle_color_bar = COLOR_BAR.copy()
    random.shuffle(shuffle_color_bar)
    return shuffle_color_bar


def draw_2d_line(
        x: list,
        y_ls: list,
        save_path: str,
        label_ls: list | None = None,
        color_ls: list | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        grid: bool = True,
        dpi: int = 500
):
    plt.figure()
    plt.box(True)
    plt.grid(grid)
    y_ls = y_ls if isinstance(y_ls[0], (list, tuple)) else [y_ls]
    color_ls = COLOR_BAR if color_ls is None else color_ls  # random_color_bar()

    for idx, y in enumerate(y_ls):
        plt.plot(x, y, color=color_ls[idx], marker="o", markerfacecolor=color_ls[idx],
                 label=label_ls[idx] if label_ls is not None else None, zorder=2)

    if label_ls is not None:
        plt.legend(loc='best')
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.savefig(os.path.join(save_path), dpi=dpi)


def draw_bin_image(
        info: list,
        save_path: str | None = None,
        show: bool = False,
        bins: int = 100,
        title: str = '',
        cumu_show: bool = True,
        density: bool = True,
        annotation: bool = False,
        cumu_thred_ls: list | tuple = (0.5, 0.95),
        save_dpi: int = 300,
        **hist_params
):
    """
    draw bin with cumulative curve
    Args:

    """

    def _draw(_cumu_thred_ls, _base_ratio_zip, _max_value, _annotation):
        COLOR_BAR = ["red", "blue", "green", "pink"]
        for idx, c_thred in enumerate(_cumu_thred_ls):
            x_loc = [i[0] for i in _base_ratio_zip if i[1] >= c_thred][0]

            plt.axvline(
                x_loc,
                color=COLOR_BAR[idx % (COLOR_BAR.__len__())],
                linestyle='-.',
                linewidth=2,
                label=f'{c_thred}%'
            )
            if _annotation:
                plt.annotate(
                    f'{x_loc:.2f}',
                    xy=(x_loc, _max_value / 2),
                    xytext=(x_loc, _max_value / 3),
                    arrowprops=dict(facecolor=COLOR_BAR[idx], arrowstyle="->")
                )

    values, base, _ = plt.hist(info, bins=bins, density=density,
                               **hist_params)  # values each bin count. base: bin border
    plt.grid()

    if cumu_show:
        values = np.append(values, 0)
        max_value = np.max(values)
        max_base = np.max(base)
        cumu_value = np.cumsum(values)
        cumu_value_ratio = cumu_value / cumu_value[-1]

        assert len(base) == len(cumu_value_ratio)
        base_ratio_zip = list(zip(base, cumu_value_ratio))

        _draw(cumu_thred_ls, base_ratio_zip, max_value, annotation)
        plt.legend()

        ax_bis = plt.twinx()
        ax_bis.plot(base, cumu_value_ratio, color='darkorange', marker='o', linestyle='-', markersize=1,
                    label="Cumulative Histogram")
        ax_bis.set_ylabel('Cumulative Histogram', color='darkorange', fontdict={'size': 9})

    if title is not None:
        plt.title(title, fontdict={'size': 9})

    # plt.title(title)
    plt.tight_layout()
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path, dpi=save_dpi)
    plt.close()


def unite_test():
    canvas = np.ones((512, 512, 3), dtype=np.uint8) * 255
    bbox = [
        np.array([[20, 20], [30, 20], [30, 40], [20, 40]]),
        np.array([[50, 50], [200, 50], [200, 80], [50, 80]])]
    # bbox = [[20, 20, 30, 40], [50, 50, 200, 80]]
    label = ["vertical text", "horizon"]
    draw_obj = DrawImage()
    res = draw_obj.poly_with_annotation_new(canvas, bbox, label, thickness=2)
    # res = draw_obj.rectangle_with_annotation_new(canvas, bbox, label)
    Image.fromarray(res).show()


if __name__ == "__main__":
    unite_test()
