# Fly0 Tested Environment

## Reference Environment

| Component | Tested version |
|---|---|
| Operating system | Windows 11 Pro 10.0.22631 |
| Unreal Engine | 4.27.2 |
| AirSim Python client | 1.8.1 |
| Python | 3.8.7, 64-bit |
| Ollama | 0.33.2 |
| Vision-language model | qwen3-vl:4b |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| GPU memory | 8188 MiB |
| NVIDIA driver | 591.86 |
| NumPy | 1.24.4 |
| SciPy | 1.10.1 |
| OpenCV Contrib | 4.7.0.68 |
| Pillow | 10.4.0 |
| Requests | 2.32.4 |
| msgpack-python | 0.5.6 |
| msgpack-rpc-python | 0.4.1 |
| Tornado | 4.5.3 |

## AirSim Verification

Verified with:

```powershell
I:\fly0_py38\Scripts\python.exe -c "import airsim; print(airsim.__version__); print(airsim.__file__)"

Expected output:

```text
1.8.1
I:\fly0_py38\lib\site-packages\airsim\__init__.py
```

## Important Note

The AirSim Python module is importable and functional in the tested Fly0
environment, but this environment does not contain standard pip distribution
metadata for AirSim. Therefore, `pip show airsim` may report that the package
is not installed even though `import airsim` succeeds.

The verified runtime interpreter is:

```text
I:\fly0_py38\Scripts\python.exe
```

For a clean contributor environment, install AirSim with:

```powershell
python -m pip install airsim==1.8.1 --no-build-isolation
```

Other Python, AirSim, Unreal Engine, or dependency versions have not yet been
validated for identical behavior.

## Runtime Requirements

- UE4 AirSim scene must be running.
- AirSim RPC must be available at `127.0.0.1:41451`.
- Ollama must be available at `http://127.0.0.1:11434`.
- The model `qwen3-vl:4b` must be installed.
- Fly0 uses AirSim NED coordinates internally: smaller Z means higher altitude.
