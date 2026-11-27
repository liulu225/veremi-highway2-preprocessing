#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
highway_2 攻击分布可视化脚本

基于 highway2_数据处理脚本.py 生成的主表，为 15 类攻击绘制
"攻击消息 vs 同攻击子集内正常消息"的分布对比图：

- 位置/速度/航向/加速度等连续字段：KDE 平滑密度曲线；
- 速度为单点值的攻击（zeroSpeedReport、suddenStop 等）：直方图；
- dosAttack / trafficCongestionSybil：数量类统计图；
- 额外输出每 sender 速率补充图与 15 合 1 总览图。

运行示例：
python plot_attack_distributions.py --input highway2_机器学习主表.parquet --output-dir 攻击分布对比

依赖：numpy pandas scipy matplotlib pillow，以及 fastparquet 或 pyarrow 之一。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image

C_BLUE, C_RED = "#4C9AFF", "#F2542D"


def setup_chinese_font():
    """尽量找到系统里的中文字体，避免标题变成方块。"""
    from matplotlib import font_manager
    candidates = ["Microsoft YaHei", "SimHei", "PingFang SC",
                  "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def read_main_table(path):
    cols = ["source_attack_subset", "is_attack", "sender_spd", "sender_hed",
            "sender_acl", "sender_pos_x", "sender_pos_y", "delay_ns",
            "rcvTime", "sender_id"]
    try:
        return pd.read_parquet(path, columns=cols)
    except Exception:
        return pd.read_parquet(path, columns=cols, engine="fastparquet")


def _finish_ax(ax, title, xlabel):
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("密度（已归一化）")
    ax.legend()
    ax.grid(alpha=0.25)


class AttackPlotter:
    def __init__(self, df, out_dir):
        self.df = df
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.summaries = []

    def groups(self, subset, col):
        sub = self.df[self.df["source_attack_subset"] == subset]
        a = sub.loc[sub["is_attack"] == 1, col].dropna().astype(float)
        b = sub.loc[sub["is_attack"] == 0, col].dropna().astype(float)
        return a, b

    @staticmethod
    def _kde(values, grid, cap=60000):
        if len(values) > cap:
            values = values.sample(cap, random_state=42)
        return gaussian_kde(values)(grid)

    def kde_compare(self, subset, col, xlabel, xlim, fname, scale=1.0):
        a, b = self.groups(subset, col)
        a, b = a * scale, b * scale
        grid = np.linspace(xlim[0], xlim[1], 700)
        ya, yb = self._kde(a, grid), self._kde(b, grid)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(grid, yb, color=C_BLUE, lw=2, label=f"正常消息（n={len(b):,}）")
        ax.fill_between(grid, yb, color=C_BLUE, alpha=0.15)
        ax.plot(grid, ya, color=C_RED, lw=2, label=f"攻击消息（n={len(a):,}）")
        ax.fill_between(grid, ya, color=C_RED, alpha=0.15)
        ax.axvline(b.median(), color=C_BLUE, ls="--", lw=1.2, alpha=0.8)
        ax.axvline(a.median(), color=C_RED, ls="--", lw=1.2, alpha=0.8)
        ax.text(0.98, 0.72,
                f"中位数：攻击 {a.median():.2f} / 正常 {b.median():.2f}\n差值 ≈ {a.median()-b.median():+.2f}",
                transform=ax.transAxes, ha="right", fontsize=10,
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
        clipped = int(((a > xlim[1]) | (a < xlim[0])).sum()
                      + ((b > xlim[1]) | (b < xlim[0])).sum())
        if clipped:
            ax.text(0.98, 0.03, f"另有 {clipped} 条超出坐标范围未显示",
                    transform=ax.transAxes, ha="right", fontsize=9, color="gray")
        ax.set_xlim(*xlim)
        _finish_ax(ax, f"{subset}：{xlabel}分布对比（攻击 vs 同包正常）· 全量", xlabel)
        fig.tight_layout()
        fig.savefig(self.out / fname, dpi=150)
        plt.close(fig)
        return f"{subset}: 攻击n={len(a)} 正常n={len(b)} 中位数 {a.median():.2f}/{b.median():.2f}"

    def hist_compare(self, subset, col, xlabel, xlim, fname):
        a, b = self.groups(subset, col)
        bins = np.arange(xlim[0], xlim[1] + 1, 1.0)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.hist(b, bins=bins, density=True, alpha=0.55, color=C_BLUE, label=f"正常消息（n={len(b):,}）")
        ax.hist(a, bins=bins, density=True, alpha=0.65, color=C_RED, label=f"攻击消息（n={len(a):,}）")
        ax.axvline(b.median(), color=C_BLUE, ls="--", lw=1.2, alpha=0.8)
        ax.axvline(a.median(), color=C_RED, ls="--", lw=1.2, alpha=0.8)
        ax.text(0.98, 0.72,
                f"中位数：攻击 {a.median():.2f} / 正常 {b.median():.2f}\n差值 ≈ {a.median()-b.median():+.2f}",
                transform=ax.transAxes, ha="right", fontsize=10,
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
        clipped = int((np.concatenate([a, b]) > xlim[1]).sum())
        if clipped:
            ax.text(0.98, 0.03, f"另有 {clipped} 条 > {xlim[1]} 未显示",
                    transform=ax.transAxes, ha="right", fontsize=9, color="gray")
        ax.set_xlim(*xlim)
        _finish_ax(ax, f"{subset}：{xlabel}分布对比（攻击 vs 同包正常）· 全量·直方图", xlabel)
        fig.tight_layout()
        fig.savefig(self.out / fname, dpi=150)
        plt.close(fig)
        return f"{subset}: 攻击n={len(a)} 正常n={len(b)} 中位数 {a.median():.2f}/{b.median():.2f} [直方图]"

    def smart_compare(self, subset, col, xlabel, xlim, fname, scale=1.0):
        a, b = self.groups(subset, col)
        if a.std() < 1e-9 or b.std() < 1e-9 or a.nunique() < 5 or b.nunique() < 5:
            return self.hist_compare(subset, col, xlabel, xlim, fname)
        return self.kde_compare(subset, col, xlabel, xlim, fname, scale=scale)

    def pos_compare(self, subset, fname):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        for ax, col, center, label in [
            (axes[0], "sender_pos_x", 216000, "pos_x − 216000（米）"),
            (axes[1], "sender_pos_y", 449000, "pos_y − 449000（米）"),
        ]:
            a, b = self.groups(subset, col)
            a, b = a - center, b - center
            both = np.concatenate([a, b])
            lo = max(0, np.quantile(both, 0.001) - 20)
            hi = np.quantile(both, 0.999) + 20
            grid = np.linspace(lo, hi, 600)
            ya, yb = self._kde(a, grid), self._kde(b, grid)
            ax.plot(grid, yb, color=C_BLUE, lw=2, label=f"正常（n={len(b):,}）")
            ax.fill_between(grid, yb, color=C_BLUE, alpha=0.15)
            ax.plot(grid, ya, color=C_RED, lw=2, label=f"攻击（n={len(a):,}）")
            ax.fill_between(grid, ya, color=C_RED, alpha=0.15)
            ax.axvline(b.median(), color=C_BLUE, ls="--", lw=1.1, alpha=0.8)
            ax.axvline(a.median(), color=C_RED, ls="--", lw=1.1, alpha=0.8)
            ax.set_title(f"{label}：中位数差 {a.median()-b.median():+.1f} 米", fontsize=11)
            ax.set_xlabel(label)
            ax.set_ylabel("密度")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.25)
        fig.suptitle(f"{subset}：sender 位置分布对比（攻击 vs 同包正常）· 全量", fontsize=13)
        fig.tight_layout()
        fig.savefig(self.out / fname, dpi=150)
        plt.close(fig)
        ax_, bx_ = self.groups(subset, "sender_pos_x")
        ay_, by_ = self.groups(subset, "sender_pos_y")
        return (f"{subset}: 位置中位数差 x={ax_.median()-bx_.median():+.1f}米 "
                f"y={ay_.median()-by_.median():+.1f}米")

    def dos_time_series(self):
        sub = self.df[self.df["source_attack_subset"] == "dosAttack"]
        t0 = sub["rcvTime"].min()
        a_t = (sub.loc[sub["is_attack"] == 1, "rcvTime"] - t0) / 1e9
        b_t = (sub.loc[sub["is_attack"] == 0, "rcvTime"] - t0) / 1e9
        binw = 10
        tbins = np.arange(0, max(a_t.max(), b_t.max()) + binw, binw)
        ca, _ = np.histogram(a_t, bins=tbins)
        cb, _ = np.histogram(b_t, bins=tbins)
        tc = (tbins[:-1] + tbins[1:]) / 2
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(tc, cb, color=C_BLUE, lw=2, label=f"正常消息（共 {len(b_t):,} 条）")
        ax.plot(tc, ca, color=C_RED, lw=2, label=f"攻击消息（共 {len(a_t):,} 条）")
        ax.fill_between(tc, ca, color=C_RED, alpha=0.15)
        _finish_ax(ax, "dosAttack：每 10 秒消息数（攻击 vs 同包正常）· 全量", "时间（秒）")
        ax.set_ylabel("每 10 秒消息条数")
        fig.tight_layout()
        fig.savefig(self.out / "dosAttack.png", dpi=150)
        plt.close(fig)

    def sender_rate_compare(self, subset, fname):
        sub = self.df[self.df["source_attack_subset"] == subset].copy()
        sub["tsec"] = ((sub["rcvTime"] - sub["rcvTime"].min()) / 1e9 / 10).astype(int)
        a = sub[sub["is_attack"] == 1].groupby(["tsec", "sender_id"]).size().astype(float)
        b = sub[sub["is_attack"] == 0].groupby(["tsec", "sender_id"]).size().astype(float)
        cap = float(np.quantile(np.concatenate([a.values, b.values]), 0.995))
        grid = np.linspace(0, cap, 600)
        ya = gaussian_kde(a)(grid) if a.std() > 0 else np.zeros_like(grid)
        yb = gaussian_kde(b)(grid)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(grid, yb, color=C_BLUE, lw=2, label=f"正常 sender（{len(b):,} 个 sender×时间窗）")
        ax.fill_between(grid, yb, color=C_BLUE, alpha=0.15)
        ax.plot(grid, ya, color=C_RED, lw=2, label=f"攻击 sender（{len(a):,} 个 sender×时间窗）")
        ax.fill_between(grid, ya, color=C_RED, alpha=0.15)
        ax.axvline(b.median(), color=C_BLUE, ls="--", lw=1.2, alpha=0.8)
        ax.axvline(a.median(), color=C_RED, ls="--", lw=1.2, alpha=0.8)
        ax.set_xlim(0, cap)
        _finish_ax(ax, f"{subset}：每 10 秒每个 sender 的消息数分布（攻击 vs 同包正常）· 全量",
                   "每 10 秒每个 sender 的消息条数")
        fig.tight_layout()
        fig.savefig(self.out / fname, dpi=150)
        plt.close(fig)
        return (f"{subset}: 每10秒每sender消息数 攻击中位数={a.median():.0f} "
                f"正常中位数={b.median():.0f}")

    def sybil_compare(self):
        sub = self.df[self.df["source_attack_subset"] == "trafficCongestionSybil"].copy()
        sub["tsec"] = ((sub["rcvTime"] - sub["rcvTime"].min()) / 1e9 / 10).astype(int)
        a_sy = sub[sub["is_attack"] == 1].groupby("tsec")["sender_id"].nunique()
        b_sy = sub[sub["is_attack"] == 0].groupby("tsec")["sender_id"].nunique()
        fig, ax = plt.subplots(figsize=(9, 5.5))
        bins = np.arange(0, int(max(a_sy.max(), b_sy.max())) + 2) - 0.5
        ax.hist(b_sy, bins=bins, density=True, alpha=0.55, color=C_BLUE,
                label=f"正常消息（{len(b_sy)} 个时间窗）")
        ax.hist(a_sy, bins=bins, density=True, alpha=0.65, color=C_RED,
                label=f"攻击消息（{len(a_sy)} 个时间窗）")
        _finish_ax(ax, "trafficCongestionSybil：每 10 秒不同 sender 数量分布（攻击 vs 同包正常）· 全量",
                   "每 10 秒时间窗内不同 sender_id 的数量")
        fig.tight_layout()
        fig.savefig(self.out / "trafficCongestionSybil.png", dpi=150)
        plt.close(fig)
        return (f"trafficCongestionSybil: 每10秒不同sender数 攻击中位数={a_sy.median():.0f} "
                f"正常中位数={b_sy.median():.0f}")

    def make_overview(self, fname="_总览.png"):
        names = ["constantPositionOffset", "randomPositionOffset", "positionMirroring",
                 "constantSpeedOffset", "randomSpeedOffset", "suddenConstantSpeed",
                 "zeroSpeedReport", "suddenStop", "reversedHeading",
                 "feignedBraking", "accelerationMultiplication", "timeDelayAttack",
                 "dataReplay", "dosAttack_每sender速率", "trafficCongestionSybil"]
        thumb_w = 640
        thumbs = []
        for n in names:
            im = Image.open(self.out / f"{n}.png")
            thumbs.append(im.resize((thumb_w, int(im.height * thumb_w / im.width))))
        cell_h = max(t.height for t in thumbs)
        sheet = Image.new("RGB", (3 * thumb_w, 5 * cell_h), "white")
        for i, t in enumerate(thumbs):
            sheet.paste(t, ((i % 3) * thumb_w, (i // 3) * cell_h))
        sheet.save(self.out / fname)


def main():
    parser = argparse.ArgumentParser(description="highway_2 攻击分布可视化")
    parser.add_argument("--input", required=True, help="主表 parquet 路径")
    parser.add_argument("--output-dir", default="攻击分布对比", help="图片输出目录")
    args = parser.parse_args()

    setup_chinese_font()
    df = read_main_table(args.input)
    print("主表读取:", df.shape, flush=True)

    p = AttackPlotter(df, args.output_dir)
    p.summaries.append(p.kde_compare("constantSpeedOffset", "sender_spd", "sender 速度（m/s）", (0, 35), "constantSpeedOffset.png"))
    p.summaries.append(p.kde_compare("randomSpeedOffset", "sender_spd", "sender 速度（m/s）", (0, 35), "randomSpeedOffset.png"))
    p.summaries.append(p.smart_compare("suddenStop", "sender_spd", "sender 速度（m/s）", (0, 35), "suddenStop.png"))
    p.summaries.append(p.smart_compare("reversedHeading", "sender_hed", "sender 航向角（度）", (0, 360), "reversedHeading.png"))
    p.summaries.append(p.smart_compare("feignedBraking", "sender_acl", "sender 加速度（m/s²）", (-10, 10), "feignedBraking.png"))
    p.summaries.append(p.smart_compare("accelerationMultiplication", "sender_acl", "sender 加速度（m/s²）", (-10, 10), "accelerationMultiplication.png"))
    p.summaries.append(p.smart_compare("timeDelayAttack", "delay_ns", "收发时间差 delay（毫秒）", (0.8, 3.2), "timeDelayAttack.png", scale=1e-6))
    p.summaries.append(p.smart_compare("dataReplay", "sender_spd", "sender 速度（m/s）", (0, 35), "dataReplay.png"))
    p.summaries.append(p.smart_compare("zeroSpeedReport", "sender_spd", "sender 速度（m/s）", (0, 35), "zeroSpeedReport.png"))
    p.summaries.append(p.smart_compare("suddenConstantSpeed", "sender_spd", "sender 速度（m/s）", (0, 35), "suddenConstantSpeed.png"))
    for s in ["constantPositionOffset", "randomPositionOffset", "positionMirroring"]:
        p.summaries.append(p.pos_compare(s, f"{s}.png"))
    p.dos_time_series()
    p.summaries.append(p.sender_rate_compare("dosAttack", "dosAttack_每sender速率.png"))
    p.summaries.append(p.sender_rate_compare("timeDelayAttack", "timeDelayAttack_每sender速率.png"))
    p.summaries.append(p.sybil_compare())
    p.make_overview()

    summary_path = Path(args.output_dir) / "绘图摘要.txt"
    summary_path.write_text("\n".join(p.summaries), encoding="utf-8")
    print("\n".join(p.summaries))
    print("全部完成，输出目录:", args.output_dir)


if __name__ == "__main__":
    main()
