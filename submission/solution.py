import subprocess
import sys

# ── Environment setup ────────────────────────────────────────────────────────

def set_env(input_archive, temp_dir):
    import os
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        subprocess.run(['tar', '-xzf', input_archive, '-C', temp_dir], check=True)
    subprocess.run([
        sys.executable, '-m', 'pip', 'install',
        '--no-index', '--find-links', f'{temp_dir}/wheels',
        'unsloth', 'trl', 'vllm', 'openai_harmony'
    ], check=True)


set_env(
    input_archive='/kaggle/input/aimo-3-utils/wheels.tar.gz',
    temp_dir='/kaggle/tmp/setup'
)

subprocess.run(['ls', '/kaggle/tmp/setup/tiktoken_encodings'])

# ── Env vars ─────────────────────────────────────────────────────────────────

import os

os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda/bin/ptxas'
os.environ['TIKTOKEN_ENCODINGS_BASE'] = '/kaggle/tmp/setup/tiktoken_encodings'

# ── Imports ───────────────────────────────────────────────────────────────────

import warnings
warnings.simplefilter('ignore')

import gc
import re
import math
import time
import queue
import threading
import contextlib
from typing import Optional
from jupyter_client import KernelManager
from collections import Counter, defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor

import pandas as pd
import polars as pl

from openai import OpenAI

from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    SystemContent,
    ReasoningEffort,
    ToolNamespaceConfig,
    Author,
    Message,
    Role,
    TextContent,
    Conversation,
)

from transformers import set_seed
import kaggle_evaluation.aimo_3_inference_server

# ── Config ────────────────────────────────────────────────────────────────────

class CFG:

    system_prompt = (
        'You are an elite mathematical problem solver with expertise at the International '
        'Mathematical Olympiad (IMO) level. Your goal is to find the correct answer through '
        'rigorous mathematical reasoning.\n\n'
        'More background: problems spanning algebra, combinatorics, geometry, and number theory.\n\n'

        '# Problem-Solving Approach:\n'
        '1. UNDERSTAND: Carefully read and rephrase the problem in your own words. '
        'Identify what is given, what needs to be found, and any constraints.\n'
        '2. Restate & parse: Restate the problem in your own words. List all given constraints, '
        'what is asked, and the domain (integers/reals/geometry/etc.).\n'
        '3. EXPLORE: Consider multiple solution strategies. Think about relevant theorems, '
        'techniques, patterns, or analogous problems. Don\'t commit to one approach immediately.\n'
        '4. PLAN: Select the most promising approach and outline key steps before executing.\n'
        '5. Choose a representation: Introduce variables for all unknowns. Rewrite the conditions '
        'as equations/inequalities. If there is a transformation that simplifies structure '
        '(substitution, shift, factorization, symmetry), do it and explain why.\n'
        '6. EXECUTE: Work through your solution methodically. Show all reasoning steps clearly. '
        'Look for quantities that stay constant (sum/product/parity/mod, angle chasing invariants, '
        'potential functions), or monotonic behavior enabling bounds. State the key invariant clearly.\n'
        '7. VERIFY: Check your answer by substituting back, testing edge cases, or using '
        'alternative methods. Ensure logical consistency throughout.\n\n'

        '# Mathematical Reasoning Principles:\n'
        '- Break complex problems into smaller, manageable sub-problems\n'
        '- Look for patterns, symmetries, and special cases that provide insight\n'
        '- Use concrete examples to build intuition before generalizing\n'
        '- If there are similar problems you saw before, try its solution, it might work\n'
        '- Consider extreme cases and boundary conditions. If the problem naturally branches, '
        'enumerate cases cleanly, solve each, and discard impossible ones using constraints.\n'
        '- If stuck, try working backwards from the desired result\n'
        '- Be willing to restart with a different approach if needed\n\n'

        '# Verification Requirements:\n'
        '- Cross-check arithmetic and algebraic manipulations\n'
        '- Verify that your solution satisfies all problem constraints\n'
        '- Test your answer with simple cases or special values when possible\n'
        '- Ensure dimensional consistency and reasonableness of the result\n\n'

        '# Output Format:\n'
        'The final answer must be a non-negative integer between 0 and 99999.\n'
        'Place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}\n\n'

        'Think step-by-step and show your complete reasoning process. Quality of reasoning '
        'is as important as the final answer.'
    )

    tool_prompt = (
        'Use this tool to execute Python code for:\n'
        '- Complex calculations that would be error-prone by hand\n'
        '- Numerical verification of analytical results\n'
        '- Generating examples or testing conjectures\n'
        '- Visualizing problem structure when helpful\n'
        '- Brute-force verification for small cases\n\n'

        'The environment is a stateful Jupyter notebook. Code persists between executions.\n'
        'Always use print() to display results. Write clear, well-commented code.\n\n'

        'Remember: Code should support your mathematical reasoning, not replace it. '
        'Explain what you\'re computing and why before running code.'
    )

    preference_prompt = (
        'You have access to `math`, `numpy`, and `sympy` for:\n\n'

        '# Symbolic Computation (sympy):\n'
        '- Algebraic manipulation and simplification\n'
        '- Solving equations and systems of equations\n'
        '- Symbolic differentiation and integration\n'
        '- Number theory functions (primes, divisors, modular arithmetic)\n'
        '- Polynomial operations and factorization\n'
        '- Working with mathematical expressions symbolically\n\n'

        '# Numerical Computation (numpy):\n'
        '- Array operations and linear algebra\n'
        '- Efficient numerical calculations for large datasets\n'
        '- Matrix operations and eigenvalue problems\n'
        '- Statistical computations\n\n'

        '# Mathematical Functions (math):\n'
        '- Standard mathematical functions (trig, log, exp)\n'
        '- Constants like pi and e\n'
        '- Basic operations for single values\n\n'

        'Best Practices:\n'
        '- Use sympy for exact symbolic answers when possible\n'
        '- Use numpy for numerical verification and large-scale computation\n'
        '- Combine symbolic and numerical approaches: derive symbolically, verify numerically\n'
        '- Document your computational strategy clearly\n'
        '- Validate computational results against known cases or theoretical bounds'
    )

    computational_prompt = (
        'You are an elite mathematical problem solver. '
        'Your primary strategy is computational: use Python extensively to enumerate, brute-force, '
        'and verify every claim.\n\n'
        '# Approach:\n'
        '1. Parse the problem and immediately identify what can be computed directly.\n'
        '2. Write Python code to brute-force small cases and reveal patterns.\n'
        '3. Use sympy for exact symbolic answers and numpy for numerical checks.\n'
        '4. Never commit to an answer without programmatic verification.\n\n'
        '# Output Format:\n'
        'The final answer must be a non-negative integer between 0 and 99999.\n'
        'Place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}\n\n'
        'Prioritize Python computation and verification over manual derivation.'
    )

    backward_prompt = (
        'You are an elite mathematical problem solver. '
        'Your approach emphasizes constraint verification and working backwards.\n\n'
        '# Approach:\n'
        '1. List every constraint the answer must satisfy.\n'
        '2. Work backwards: what properties must the answer have?\n'
        '3. Eliminate impossible values systematically using constraints.\n'
        '4. Verify your final answer satisfies ALL stated conditions before boxing it.\n\n'
        '# Output Format:\n'
        'The final answer must be a non-negative integer between 0 and 99999.\n'
        'Place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}\n\n'
        'Always explicitly verify your answer satisfies every constraint before finalizing.'
    )

    temperatures = [0.7, 1.0, 1.0, 1.2, 1.0, 0.7, 1.0, 1.2, 1.0, 1.0]

    served_model_name = 'gpt-oss'
    model_path = '/kaggle/input/gpt-oss-120b/transformers/default/1'

    kv_cache_dtype = 'fp8_e4m3'
    dtype = 'auto'

    # Timeout configs
    high_problem_timeout = 900
    base_problem_timeout = 300
    notebook_limit = 30000  # ~8.3 hrs solving time; ~40 min buffer for startup within the 9-hr wall clock
    server_timeout = 180
    session_timeout = 960
    jupyter_timeout = 30  # raised from 6s — allows heavier sympy/numpy computations to complete
    sandbox_timeout = 3

    stream_interval = 200
    context_tokens = 65536
    buffer_tokens = 512
    search_tokens = 32
    top_logprobs = 5
    batch_size = 256
    early_stop = 6
    attempts = 10
    workers = 16
    turns = 128
    seed = 42

    gpu_memory_utilization = 0.96
    temperature = 1
    min_p = 0.02

    # Verifier config
    verifier_temperature = 0.3
    verifier_max_tokens = 1024


set_seed(CFG.seed)

# ── Template ──────────────────────────────────────────────────────────────────

class AIMO3Template:

    def __init__(self):
        pass

    def get_system_content(self, system_prompt: str, tool_config: ToolNamespaceConfig) -> SystemContent:
        return (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )

    def apply_chat_template(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_config: ToolNamespaceConfig,
    ) -> list[Message]:
        system_content = self.get_system_content(system_prompt, tool_config)
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
        user_message = Message.from_role_and_content(Role.USER, user_prompt)
        return [system_message, user_message]


# ── Sandbox ───────────────────────────────────────────────────────────────────

class AIMO3Sandbox:

    _port_lock = threading.Lock()
    _start_lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def _get_next_ports(cls, count: int = 5) -> list[int]:
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count
            return ports

    def __init__(self, timeout: float):
        self._default_timeout = timeout
        self._owns_kernel = False
        self._client = None
        self._km = None

        ports = self._get_next_ports(5)

        env = os.environ.copy()
        env['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
        env['PYDEVD_WARN_EVALUATION_TIMEOUT'] = '0'
        env['JUPYTER_PLATFORM_DIRS'] = '1'
        env['PYTHONWARNINGS'] = 'ignore'
        env['MPLBACKEND'] = 'Agg'

        self._km = KernelManager()
        self._km.shell_port = ports[0]
        self._km.iopub_port = ports[1]
        self._km.stdin_port = ports[2]
        self._km.hb_port = ports[3]
        self._km.control_port = ports[4]

        with self.__class__._start_lock:
            self._km.start_kernel(env=env, extra_arguments=['--Application.log_level=CRITICAL'])
            self._client = self._km.blocking_client()
            self._client.start_channels()
            self._client.wait_for_ready(timeout=self._default_timeout)
        self._owns_kernel = True

        self.execute(
            'import math\n'
            'import fractions\n'
            'import numpy\n'
            'import scipy\n'
            'import sympy\n'
            'import networkx\n'
            'import itertools\n'
            'import collections\n'
            'import mpmath\n'
            'mpmath.mp.dps = 64\n'
        )

    def _format_error(self, traceback: list[str]) -> str:
        clean_lines = []
        for frame in traceback:
            clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)
            if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
                continue
            clean_lines.append(clean_frame)
        return ''.join(clean_lines)

    def execute(self, code: str, timeout: float | None = None) -> str:
        client = self._client
        effective_timeout = timeout or self._default_timeout

        msg_id = client.execute(code, store_history=True, allow_stdin=False, stop_on_error=False)

        stdout_parts = []
        stderr_parts = []
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > effective_timeout:
                self._km.interrupt_kernel()
                return f'[ERROR] Execution timed out after {effective_timeout} seconds'

            try:
                msg = client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue

            if msg.get('parent_header', {}).get('msg_id') != msg_id:
                continue

            msg_type = msg.get('msg_type')
            content = msg.get('content', {})

            if msg_type == 'stream':
                text = content.get('text', '')
                if content.get('name') == 'stdout':
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)

            elif msg_type == 'error':
                stderr_parts.append(self._format_error(content.get('traceback', [])))

            elif msg_type in {'execute_result', 'display_data'}:
                text = content.get('data', {}).get('text/plain')
                if text:
                    stdout_parts.append(text if text.endswith('\n') else f'{text}\n')

            elif msg_type == 'status':
                if content.get('execution_state') == 'idle':
                    break

        stdout = ''.join(stdout_parts)
        stderr = ''.join(stderr_parts)

        if stderr:
            return f'{stdout.rstrip()}\n{stderr}' if stdout else stderr

        return stdout if stdout.strip() else '[WARN] No output. Use print() to see results.'

    def close(self):
        with contextlib.suppress(Exception):
            if self._client:
                self._client.stop_channels()
        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)
            with contextlib.suppress(Exception):
                self._km.cleanup_resources()

    def reset(self):
        self.execute(
            '%reset -f\n'
            'import math\n'
            'import numpy\n'
            'import sympy\n'
            'import itertools\n'
            'import collections\n'
            'import mpmath\n'
            'mpmath.mp.dps = 64\n'
        )

    def __del__(self):
        self.close()


# ── Tool ──────────────────────────────────────────────────────────────────────

class AIMO3Tool:

    def __init__(self, local_jupyter_timeout: float, tool_prompt: str, sandbox=None):
        self._local_jupyter_timeout = local_jupyter_timeout
        self._tool_prompt = tool_prompt
        self._jupyter_session = sandbox
        self._owns_session = sandbox is None
        self._execution_lock = threading.Lock()
        self._init_lock = threading.Lock()

    def _ensure_session(self):
        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = AIMO3Sandbox(timeout=self._local_jupyter_timeout)

    def _ensure_last_print(self, code: str) -> str:
        lines = code.strip().split('\n')
        if not lines:
            return code
        last_line = lines[-1].strip()
        if 'print' in last_line or 'import' in last_line or not last_line or last_line.startswith('#'):
            return code
        lines[-1] = 'print(' + last_line + ')'
        return '\n'.join(lines)

    @property
    def instruction(self) -> str:
        return self._tool_prompt

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(name='python', description=self.instruction, tools=[])

    def _make_response(self, output: str, channel: str | None = None) -> Message:
        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name='python')
        message = Message(author=author, content=[content]).with_recipient('assistant')
        if channel:
            message = message.with_channel(channel)
        return message

    def process_sync_plus(self, message: Message) -> list[Message]:
        self._ensure_session()
        raw_script = message.content[0].text
        final_script = self._ensure_last_print(raw_script)
        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(final_script)
            except TimeoutError as exc:
                output = f'[ERROR] {exc}'
        return [self._make_response(output, channel=message.channel)]


# ── Solver ────────────────────────────────────────────────────────────────────

class AIMO3Solver:

    def __init__(self, cfg, port: int = 8000):
        self.cfg = cfg
        self.port = port
        self.base_url = f'http://0.0.0.0:{port}/v1'
        self.api_key = 'sk-local'
        self.template = AIMO3Template()
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

        self._preload_model_weights()
        self.server_process = self._start_server()

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.cfg.session_timeout,
        )

        self._wait_for_server()
        self._initialize_kernels()

        self.notebook_start_time = time.time()
        self.problems_remaining = 50

    def _preload_model_weights(self) -> None:
        print(f'Loading model weights from {self.cfg.model_path} into OS Page Cache...')
        start_time = time.time()

        files_to_load = []
        total_size = 0

        for root, _, files in os.walk(self.cfg.model_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    files_to_load.append(file_path)
                    total_size += os.path.getsize(file_path)

        def _read_file(path: str) -> None:
            with open(path, 'rb') as f:
                while f.read(1024 * 1024 * 1024):
                    pass

        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            list(executor.map(_read_file, files_to_load))

        elapsed = time.time() - start_time
        print(f'Processed {len(files_to_load)} files ({total_size / 1e9:.2f} GB) in {elapsed:.2f} seconds.\n')

    def _start_server(self) -> subprocess.Popen:
        cmd = [
            sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
            '--seed', str(self.cfg.seed),
            '--model', self.cfg.model_path,
            '--served-model-name', self.cfg.served_model_name,
            '--tensor-parallel-size', '1',
            '--max-num-seqs', str(self.cfg.batch_size),
            '--gpu-memory-utilization', str(self.cfg.gpu_memory_utilization),
            '--host', '0.0.0.0',
            '--port', str(self.port),
            '--dtype', self.cfg.dtype,
            '--kv-cache-dtype', self.cfg.kv_cache_dtype,
            '--max-model-len', str(self.cfg.context_tokens),
            '--stream-interval', str(self.cfg.stream_interval),
            '--async-scheduling',
            '--disable-log-stats',
            '--enable-prefix-caching',
        ]

        self.log_file = open('vllm_server.log', 'w')
        return subprocess.Popen(cmd, stdout=self.log_file, stderr=subprocess.STDOUT, start_new_session=True)

    def _wait_for_server(self):
        print('Waiting for vLLM server...')
        start_time = time.time()

        for _ in range(self.cfg.server_timeout):
            return_code = self.server_process.poll()

            if return_code is not None:
                self.log_file.flush()
                with open('vllm_server.log', 'r') as f:
                    logs = f.read()
                raise RuntimeError(f'Server died with code {return_code}. Full logs:\n{logs}\n')

            try:
                self.client.models.list()
                elapsed = time.time() - start_time
                print(f'Server is ready (took {elapsed:.2f} seconds).\n')
                return
            except Exception:
                time.sleep(1)

        raise RuntimeError('Server failed to start (timeout).\n')

    def _initialize_kernels(self) -> None:
        print(f'Initializing {self.cfg.workers} persistent Jupyter kernels...')
        start_time = time.time()

        self.sandbox_pool = queue.Queue()

        def _create_sandbox():
            for attempt in range(3):
                try:
                    return AIMO3Sandbox(timeout=self.cfg.jupyter_timeout)
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(0.5)

        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            futures = [executor.submit(_create_sandbox) for _ in range(self.cfg.workers)]
            for future in as_completed(futures):
                self.sandbox_pool.put(future.result())

        elapsed = time.time() - start_time
        print(f'Kernels initialized in {elapsed:.2f} seconds.\n')

    def _scan_for_answer(self, text: str) -> int | None:
        pattern = r'\\boxed\s*\{\s*([0-9,]+)\s*\}'
        matches = re.findall(pattern, text)
        if matches:
            try:
                value = int(matches[-1].replace(',', ''))
                if 0 <= value <= 99999:
                    return value
            except ValueError:
                pass

        pattern = r'final\s+answer\s+is\s*:?\s*([0-9,]+)(?!\s*(?:not|tentative|wrong|incorrect|certain))'
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                value = int(matches[-1].replace(',', ''))
                if 0 <= value <= 99999:
                    return value
            except ValueError:
                pass

        pattern = r'(?:answer|result)\s*[=:]\s*([0-9,]+)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                value = int(matches[-1].replace(',', ''))
                if 0 <= value <= 99999:
                    return value
            except ValueError:
                pass

        return None

    def _compute_stats(self, logprobs_buffer: list) -> tuple:
        if not logprobs_buffer:
            return float('inf'), float('-inf')

        total_entropy = 0.0
        total_max_logprob = 0.0
        token_count = 0

        for top_logprobs_dict in logprobs_buffer:
            if not isinstance(top_logprobs_dict, dict) or not top_logprobs_dict:
                continue

            token_entropy = 0.0
            max_lp = float('-inf')

            for token_str, log_prob in top_logprobs_dict.items():
                prob = math.exp(log_prob)
                if prob > 0:
                    token_entropy -= prob * math.log2(prob)
                if log_prob > max_lp:
                    max_lp = log_prob

            total_entropy += token_entropy
            total_max_logprob += max_lp
            token_count += 1

        if token_count == 0:
            return float('inf'), float('-inf')

        return total_entropy / token_count, total_max_logprob / token_count

    def _get_strategy_prompt(self, attempt_index: int) -> str:
        if attempt_index < 4:
            return self.cfg.system_prompt
        elif attempt_index < 7:
            return self.cfg.computational_prompt
        else:
            return self.cfg.backward_prompt

    def _verify_answer(self, problem: str, answer: int, deadline: float) -> bool:
        if time.time() > deadline - 40:
            return True

        verify_prompt = (
            f'Problem:\n{problem}\n\n'
            f'A student claims the answer is {answer}.\n\n'
            'Verify step-by-step whether this answer is correct.\n'
            'Check that it satisfies every stated constraint.\n'
            'End your response with exactly one word: CORRECT or INCORRECT.'
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.served_model_name,
                messages=[{'role': 'user', 'content': verify_prompt}],
                temperature=self.cfg.verifier_temperature,
                max_tokens=self.cfg.verifier_max_tokens,
                seed=self.cfg.seed + 99991,
            )
            text = resp.choices[0].message.content.strip().upper()
            if 'INCORRECT' in text[-60:]:
                return False
            if 'CORRECT' in text[-60:]:
                return True
            return True
        except Exception:
            return True

    def _process_attempt(
        self,
        problem: str,
        system_prompt: str,
        attempt_index: int,
        stop_event: threading.Event,
        deadline: float,
        temperature: float = 1.0,
    ) -> dict:
        if stop_event.is_set() or time.time() > deadline:
            return {
                'Attempt': attempt_index + 1,
                'Answer': None,
                'Python Calls': 0,
                'Python Errors': 0,
                'Response Length': 0,
                'Entropy': float('inf'),
            }

        local_tool = None
        sandbox = None
        python_calls = 0
        python_errors = 0
        total_tokens = 0
        final_answer = None
        logprobs_buffer = []
        all_text_chunks = []

        attempt_seed = (self.cfg.seed + attempt_index * 1337) % (2 ** 31)

        try:
            sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)

            local_tool = AIMO3Tool(
                local_jupyter_timeout=self.cfg.jupyter_timeout,
                tool_prompt=self.cfg.tool_prompt,
                sandbox=sandbox,
            )

            encoding = self.encoding
            messages = self.template.apply_chat_template(system_prompt, problem, local_tool.tool_config)
            conversation = Conversation.from_messages(messages)

            for _ in range(self.cfg.turns):
                if stop_event.is_set() or time.time() > deadline:
                    break

                prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
                max_tokens = self.cfg.context_tokens - len(prompt_ids)

                if max_tokens < self.cfg.buffer_tokens:
                    break

                stream = self.client.completions.create(
                    model=self.cfg.served_model_name,
                    temperature=temperature,
                    logprobs=self.cfg.top_logprobs,
                    max_tokens=max_tokens,
                    prompt=prompt_ids,
                    seed=attempt_seed,
                    stream=True,
                    extra_body={
                        'min_p': self.cfg.min_p,
                        'stop_token_ids': self.stop_token_ids,
                        'return_token_ids': True,
                    },
                )

                try:
                    token_buffer = []
                    text_chunks = []

                    for chunk in stream:
                        if stop_event.is_set() or time.time() > deadline:
                            break

                        new_tokens = chunk.choices[0].token_ids
                        new_text = chunk.choices[0].text

                        if new_tokens:
                            token_buffer.extend(new_tokens)
                            total_tokens += len(new_tokens)
                            text_chunks.append(new_text)
                            all_text_chunks.append(new_text)

                            chunk_logprobs = chunk.choices[0].logprobs
                            if chunk_logprobs is not None and chunk_logprobs.top_logprobs:
                                logprobs_buffer.extend(chunk_logprobs.top_logprobs)

                        if '}' in new_text:
                            search_text = ''.join(text_chunks[-self.cfg.search_tokens:])
                            answer = self._scan_for_answer(search_text)
                            if answer is not None:
                                final_answer = answer
                                break

                finally:
                    stream.close()

                if final_answer is not None:
                    break

                if not token_buffer:
                    break

                new_messages = encoding.parse_messages_from_completion_tokens(token_buffer, Role.ASSISTANT)
                conversation.messages.extend(new_messages)
                last_message = new_messages[-1]

                if last_message.channel == 'final':
                    final_answer = self._scan_for_answer(last_message.content[0].text)
                    break

                if last_message.recipient == 'python':
                    python_calls += 1
                    tool_responses = local_tool.process_sync_plus(last_message)
                    response_text = tool_responses[0].content[0].text

                    if response_text.startswith('[ERROR]') or 'Traceback' in response_text or 'Error:' in response_text:
                        python_errors += 1

                    conversation.messages.extend(tool_responses)

            if final_answer is None and all_text_chunks:
                final_answer = self._scan_for_answer(''.join(all_text_chunks))

        except Exception:
            python_errors += 1

        finally:
            if sandbox is not None:
                sandbox.reset()
                self.sandbox_pool.put(sandbox)

        mean_entropy, mean_max_logprob = self._compute_stats(logprobs_buffer)

        return {
            'Attempt': attempt_index + 1,
            'Response Length': total_tokens,
            'Python Calls': python_calls,
            'Python Errors': python_errors,
            'Entropy': mean_entropy,
            'MeanLogProb': mean_max_logprob,
            'Answer': final_answer,
        }

    def _select_answer(
        self,
        detailed_results: list,
        problem: str | None = None,
        deadline: float | None = None,
    ) -> int:
        answer_weights = defaultdict(float)
        answer_votes = defaultdict(int)
        answer_logprobs = defaultdict(list)

        for result in detailed_results:
            answer = result['Answer']
            entropy = result['Entropy']
            mean_lp = result.get('MeanLogProb', float('-inf'))

            if answer is not None:
                weight = 1.0 / max(entropy, 0.5)
                answer_weights[answer] += weight
                answer_votes[answer] += 1
                if mean_lp != float('-inf'):
                    answer_logprobs[answer].append(mean_lp)

        if not answer_weights:
            print('\nFinal Answer: 0\n')
            return 0

        answer_mean_lp = {
            ans: sum(lps) / len(lps) if lps else float('-inf')
            for ans, lps in answer_logprobs.items()
        }
        valid_lps = [v for v in answer_mean_lp.values() if v != float('-inf')]
        lp_min = min(valid_lps) if valid_lps else 0.0
        lp_range = (max(valid_lps) - lp_min) if len(valid_lps) > 1 else 1.0

        max_weight = max(answer_weights.values())
        max_votes = max(answer_votes.values())

        scored_answers = []
        for answer, total_weight in answer_weights.items():
            votes = answer_votes[answer]
            lp = answer_mean_lp.get(answer, float('-inf'))
            lp_score = (lp - lp_min) / lp_range if lp != float('-inf') and lp_range > 0 else 0.5

            hybrid_score = (
                0.50 * (votes / max_votes) +
                0.30 * (total_weight / max_weight) +
                0.20 * lp_score
            )
            scored_answers.append({'answer': answer, 'votes': votes, 'score': hybrid_score})

        scored_answers.sort(key=lambda x: x['score'], reverse=True)

        # OSS self-verifier on top-2 candidates
        if problem is not None and deadline is not None and len(scored_answers) >= 2:
            print(f'Verifying top-{min(2, len(scored_answers))} candidates...')
            for item in scored_answers[:2]:
                is_correct = self._verify_answer(problem, item['answer'], deadline)
                item['verified'] = is_correct
                if is_correct:
                    item['score'] += 0.30
                else:
                    item['score'] -= 0.50
                    print(f'  Verifier rejected answer {item["answer"]}')
            scored_answers.sort(key=lambda x: x['score'], reverse=True)

        vote_data = [(item['answer'], item['votes'], round(item['score'], 3)) for item in scored_answers]
        vote_dataframe = pd.DataFrame(vote_data, columns=['Answer', 'Votes', 'Score'])
        print(vote_dataframe.to_string(index=False))

        final_answer = scored_answers[0]['answer']
        print(f'\nFinal Answer: {final_answer}\n')
        return final_answer

    def solve_problem(self, problem: str) -> int:
        print(f'\nProblem: {problem}\n')

        user_input = f'{problem} {self.cfg.preference_prompt}'

        elapsed_global = time.time() - self.notebook_start_time
        time_left = self.cfg.notebook_limit - elapsed_global
        problems_left_others = max(0, self.problems_remaining - 1)
        reserved_time = problems_left_others * self.cfg.base_problem_timeout

        budget = time_left - reserved_time
        budget = min(budget, self.cfg.high_problem_timeout)
        budget = max(budget, self.cfg.base_problem_timeout)

        deadline = time.time() + budget
        print(f'Budget: {budget:.2f} seconds | Deadline: {deadline:.2f}\n')

        tasks = [
            (self._get_strategy_prompt(i), i, self.cfg.temperatures[i % len(self.cfg.temperatures)])
            for i in range(self.cfg.attempts)
        ]

        detailed_results = []
        valid_answers = []
        stop_event = threading.Event()
        executor = ThreadPoolExecutor(max_workers=self.cfg.workers)

        try:
            futures = [
                executor.submit(self._process_attempt, user_input, sp, ai, stop_event, deadline, temp)
                for sp, ai, temp in tasks
            ]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    detailed_results.append(result)

                    if result['Answer'] is not None:
                        valid_answers.append(result['Answer'])

                    counts = Counter(valid_answers).most_common(1)
                    if counts and counts[0][1] >= self.cfg.early_stop:
                        stop_event.set()
                        for f in futures:
                            f.cancel()
                        break

                except Exception as exc:
                    print(f'Future failed: {exc}')

        finally:
            stop_event.set()
            executor.shutdown(wait=True, cancel_futures=True)
            self.problems_remaining = max(0, self.problems_remaining - 1)

        # Refinement round if confidence is low
        counts = Counter(valid_answers).most_common(1)
        top_votes = counts[0][1] if counts else 0
        top_candidate = counts[0][0] if counts else None

        if top_votes < 4 and top_candidate is not None and time.time() < deadline - 120:
            print(f'\nLow confidence ({top_votes} votes). Running refinement round with hint...')

            hint_prompt = (
                self.cfg.system_prompt +
                f'\n\nNote: preliminary attempts most often suggest the answer is {top_candidate}. '
                f'Please independently verify or find the correct answer.'
            )
            refinement_tasks = [
                (hint_prompt, self.cfg.attempts + i, self.cfg.temperatures[i % len(self.cfg.temperatures)])
                for i in range(3)
            ]

            ref_stop = threading.Event()
            ref_executor = ThreadPoolExecutor(max_workers=min(3, self.cfg.workers))

            try:
                ref_futures = [
                    ref_executor.submit(self._process_attempt, user_input, sp, ai, ref_stop, deadline, temp)
                    for sp, ai, temp in refinement_tasks
                ]
                for f in as_completed(ref_futures):
                    try:
                        r = f.result()
                        detailed_results.append(r)
                        if r['Answer'] is not None:
                            valid_answers.append(r['Answer'])
                    except Exception:
                        pass
            finally:
                ref_stop.set()
                ref_executor.shutdown(wait=True, cancel_futures=True)

            print(f'Refinement added {len(refinement_tasks)} results.')

        if detailed_results:
            results_df = pd.DataFrame(detailed_results)
            results_df['Entropy'] = results_df['Entropy'].round(3)
            results_df['MeanLogProb'] = results_df['MeanLogProb'].round(3)
            results_df['Answer'] = results_df['Answer'].astype('Int64')
            print(results_df.to_string(index=False))

        if not valid_answers:
            print('\nResult: 0\n')
            return 0

        return self._select_answer(detailed_results, problem=problem, deadline=deadline)

    def __del__(self):
        if hasattr(self, 'server_process'):
            self.server_process.terminate()
            self.server_process.wait()
        if hasattr(self, 'log_file'):
            self.log_file.close()
        if hasattr(self, 'sandbox_pool'):
            while not self.sandbox_pool.empty():
                try:
                    sb = self.sandbox_pool.get_nowait()
                    sb.close()
                except Exception:
                    pass


# ── Entry point ───────────────────────────────────────────────────────────────

solver = AIMO3Solver(CFG)


def predict(id_: pl.DataFrame, question: pl.DataFrame, answer: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    id_value = id_.item(0)
    question_text = question.item(0)

    gc.disable()
    final_answer = solver.solve_problem(question_text)
    gc.enable()
    gc.collect()

    return pl.DataFrame({'id': id_value, 'answer': final_answer})


inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        ('/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv',)
    )
