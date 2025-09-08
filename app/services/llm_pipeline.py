# ==== PENTING: taruh BENAR-BENAR paling atas ====
import os
os.environ["MPLBACKEND"] = "Agg"  # render offscreen (tanpa window)
import matplotlib
matplotlib.use("Agg")
# ================================================

import json
import time
import re
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Callable

import pandas as pd

import pandasai as pai
from pandasai import SmartDataframe
from pandasai_litellm.litellm import LiteLLM
from pandasai.core.response.dataframe import DataFrameResponse
from litellm import completion

from app.core.config import settings
from app.utils.common import df_info_string

from concurrent.futures import ThreadPoolExecutor  # <-- paralel

# ====== PENTING: system content prompt di bawah ini diambil apa adanya dari skrip sumber ======
ORCHESTRATOR_SYSTEM_PROMPT = """
        Lets think step by step.
        You are a helpful assistant that is assigned to orchestrate 3 LLM PandasAI Agents that assist business data analysis.
        The role of 3 other LLM PandasAI Agents are Data Manipulator, Data Visualizer, and Data Analyser.
        You will give a specific prompt to each of those 3 LLM PandasAI Agents.
        The prompt should be a set of numbered step by step instruction of what each LLM PandasAI Agents need to do.
        The prompt should be clear, detail, and complete to not cause confusion.
        The number of instruction may differ for each LLM PandasAI Agents.
        The task example are to answer questions that the user provide such as:
        What is my revenue this week vs last week?,
        Why did my revenue drop this week?,
        Are there any surprises in my revenue metric this month?,
        Are there any notable trends in our revenue this month?,
        What is the correlation between revenue and bounces?,
        Is this conversion rate normal for this time of year?.
        You will reason your answer based on the data that the user provide.
        You are the Orchestrator. Convert a short business question into three prompts for specialists.
        All specialists operate in Python via PandasAI SmartDataframe.
        compiler_instruction will be filled with clear and detailed instruction of how to merge/compile the separate responses from the utilized agents into a single final response.
        The final response should be insightful and not just mainly raw data with no insight.
        The compiler_instruction will be given to the compiler LLM within its system content.
        Return STRICT JSON with keys: manipulator_prompt, visualizer_prompt, analyzer_prompt, compiler_instruction.
        Each value of keys should be in a single line.
        This is what the compiler user content filled with "User Prompt:{user_prompt}. Data Info:{data_info}. Data Manipulator Response:{data_manipulator_response}. Data Visualizer Response:{data_visualizer_response}. Data Analyzer Response:{data_analyzer_response}.
        The data_info data type is str, its value is from df.info().
        The data data_manipulator_response data type is pandasai.core.response.dataframe.DataFrameResponse
        The data data_visualizer_response data type is pandasai.core.response.chart.ChartResponse
        The data data_analyzer_response data type is pandasai.core.response.string.StringResponse
        Make sure data visualizer not showing empty chart
        Make sure data visualizer not overlap text
        Make sure data visualizer only shows one chart
        Make sure data manipulator not showing empty or null dataframe unless necessary
        """

def _get_content(resp: Any) -> str:
    try:
        msg = resp.choices[0].message
        return msg["content"] if isinstance(msg, dict) else msg.content
    except Exception:
        return str(resp)

def _extract_stream_chunk_text(chunk: Any) -> str:
    try:
        ch = chunk.choices[0]
        if hasattr(ch, "delta") and ch.delta:
            return ch.delta.get("content") or ""
        msg = getattr(ch, "message", None)
        if isinstance(msg, dict):
            return msg.get("content") or ""
        if msg and hasattr(msg, "content"):
            return msg.content or ""
    except Exception:
        pass
    try:
        return _get_content(chunk)
    except Exception:
        return ""

def _find_latest_png(charts_dir: Path) -> Optional[Path]:
    pngs: List[Path] = sorted(charts_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pngs[0] if pngs else None

# --- UX: render ulang bullet menjadi tabel ASCII rapi (opsional, tidak ubah jawaban model) ---
def _try_build_ascii_table(text: str) -> Optional[str]:
    pat = re.compile(
        r"^\s*-\s*(\d{4}-\d{2}):\s*Customer\s*\$([\d.,]+)\s*\|\s*Driver\s*\$([\d.,]+)\s*\|\s*Total\s*\$([\d.,]+)",
        re.IGNORECASE,
    )
    rows = []
    for line in text.splitlines():
        m = pat.match(line)
        if m:
            rows.append((m.group(1), m.group(2), m.group(3), m.group(4)))
    if not rows:
        return None

    header = ["Month", "Customer ($)", "Driver ($)", "Total ($)"]
    data = [header] + rows
    widths = [0] * 4
    for r in data:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(r):
        return " | ".join(
            str(c).rjust(widths[i]) if i > 0 else str(c).ljust(widths[i]) for i, c in enumerate(r)
        )

    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt_row(header), sep] + [fmt_row(r) for r in rows]
    return "\n".join(lines)

def run_pipeline(
    df: pd.DataFrame,
    user_prompt: str,
    charts_dir: Path,
    stream_compiler: bool = False,
    on_token: Optional[Callable[[str], None]] = None,
) -> Tuple[str, Optional[Path], Dict[str, Any]]:
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required")

    # pasang LLM ke PandasAI global + instance
    llm = LiteLLM(model=settings.LLM_MODEL, api_key=settings.OPENAI_API_KEY)
    pai.config.set({"llm": llm})

    # -- subfolder per-run supaya nggak salah ambil PNG antar job --
    charts_dir.mkdir(parents=True, exist_ok=True)
    run_dir = charts_dir / f"job_{int(time.time() * 1000)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Orchestrate (tetap)
    data_info = df_info_string(df)
    initial_response = completion(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"User Prompt:{user_prompt}\n\nData Info:\n{data_info}"},
        ],
        seed=1,
        stream=False,
        verbosity="low",
        drop_params=True,
        reasoning_effort="high",
    )
    initial_content = _get_content(initial_response)
    try:
        spec = json.loads(initial_content)
    except json.JSONDecodeError:
        start = initial_content.find("{")
        end = initial_content.rfind("}")
        spec = json.loads(initial_content[start : end + 1])

    manipulator_prompt: str = spec["manipulator_prompt"]
    visualizer_prompt: str = spec["visualizer_prompt"]
    analyzer_prompt: str = spec["analyzer_prompt"]
    compiler_instruction: str = spec["compiler_instruction"]

    # 2) Manipulator (tetap)
    dm = SmartDataframe(
        df,
        config={
            "llm": llm,
            "seed": 1,
            "stream": False,
            "verbosity": "low",
            "drop_params": True,
            "save_charts": False,
            "open_charts": False,
            "conversational": False,
            "enforce_privacy": True,
            "reasoning_effort": "low",
            "save_charts_path": str(run_dir),
            "use_error_correction_framework": False,  # percepat: no retry panjang
            "max_retries": 0,
        },
    )
    dm_resp = dm.chat(manipulator_prompt)
    df_processed = dm_resp.value if isinstance(dm_resp, DataFrameResponse) else df

    # 3) Visualizer & 4) Analyzer — dijalankan paralel (tanpa mengubah prompt)
    dv = SmartDataframe(
        df_processed,
        config={
            "llm": llm,
            "seed": 3,
            "stream": False,
            "verbosity": "low",
            "drop_params": True,
            "save_charts": True,
            "open_charts": False,
            "conversational": False,
            "enforce_privacy": True,
            "reasoning_effort": "low",
            "save_charts_path": str(run_dir),
            "use_error_correction_framework": False,
            "max_retries": 0,
        },
    )
    da = SmartDataframe(
        df_processed,
        config={
            "llm": llm,
            "seed": 1,
            "stream": False,
            "verbosity": "low",
            "drop_params": True,
            "save_charts": False,
            "open_charts": False,
            "conversational": True,
            "enforce_privacy": False,
            "reasoning_effort": "low",
            "save_charts_path": str(run_dir),
            "use_error_correction_framework": False,
            "max_retries": 0,
        },
    )
    analyzer_full_prompt = "Respond like you are communicating to a person. " + analyzer_prompt

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_vis = ex.submit(dv.chat, visualizer_prompt)
        fut_ana = ex.submit(da.chat, analyzer_full_prompt)
        dv_resp = fut_vis.result()
        da_resp = fut_ana.result()

    chart_path = _find_latest_png(run_dir)

    # 5) Compile (FINAL) — reuse data_info (tidak hitung ulang)
    final_text = ""
    compiler_messages = [
        {"role": "system", "content": compiler_instruction},  # system content tetap
        {
            "role": "user",
            "content": (
                f"User Prompt:{user_prompt}. "
                f"Data Info:{data_info}. "
                f"Data Manipulator Response:{dm_resp}. "
                f"Data Visualizer Response:{dv_resp}. "
                f"Data Analyzer Response:{da_resp}."
            ),
        },
    ]

    if not stream_compiler:
        final_response = completion(
            model=settings.LLM_MODEL,
            messages=compiler_messages,
            seed=1,
            stream=False,
            verbosity="medium",
            drop_params=True,
            reasoning_effort="low",
        )
        final_text = _get_content(final_response)
    else:
        # coba stream asli dulu
        try:
            stream_gen = completion(
                model=settings.LLM_MODEL,
                messages=compiler_messages,
                seed=1,
                stream=True,
                verbosity="medium",
                drop_params=True,
                reasoning_effort="low",
            )
            for chunk in stream_gen:
                token = _extract_stream_chunk_text(chunk)
                if token:
                    final_text += token
                    if on_token:
                        try:
                            on_token(token)
                        except Exception:
                            pass
        except Exception:
            # === Fallback: non-stream dari LLM + pseudo-stream ke UI ===
            final_response = completion(
                model=settings.LLM_MODEL,
                messages=compiler_messages,
                seed=1,
                stream=False,
                verbosity="medium",
                drop_params=True,
                reasoning_effort="low",
            )
            final_text = _get_content(final_response)
            if on_token and final_text:
                chunk_size = max(20, min(400, settings.PARTIAL_FLUSH_CHARS))
                for i in range(0, len(final_text), chunk_size):
                    try:
                        on_token(final_text[i : i + chunk_size])
                    except Exception:
                        pass

    # UX tambahan: kalau ada bullet bulanan, tampilkan tabel ASCII rapi
    ascii_tbl = _try_build_ascii_table(final_text)
    if ascii_tbl:
        addition = "\n\n```\n" + ascii_tbl + "\n```"
        final_text += addition
        if stream_compiler and on_token:
            try:
                on_token(addition)
            except Exception:
                pass

    debug = {
        "manipulator_prompt": manipulator_prompt,
        "visualizer_prompt": visualizer_prompt,
        "analyzer_prompt": analyzer_prompt,
        "compiler_instruction": compiler_instruction,
    }
    return final_text, chart_path, debug
