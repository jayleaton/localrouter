# Kandinsky-6 DiT native oracle image (linux/arm64, GB10). Built on the dev host, never on a Spark node:
#   docker buildx build --platform linux/arm64 --network none -f oracle.Dockerfile -t k6-dit-oracle:<lock12> <wheeldir>
# <wheeldir> holds the wheels from `pip download "torch==2.14.1+cu130"` (cu130 index) and lock.txt, their
# `name==version --hash=sha256:...` lines; install is offline and hash-checked, so the image pins exactly upstream's
# torch==2.14.1 on Python 3.13 (pyproject requires-python >=3.13,<3.15). The DiT twin and the upstream DiT modules
# import only torch and the stdlib. The kit is bind-mounted read-only at run time; nothing else is baked in.
FROM python@sha256:a1165e272e578941b84abc79e4ab38a0305cd12803a5c4247979ac7655f4d641
COPY . /wheels
RUN python -m pip install --no-cache-dir --no-index --find-links /wheels --require-hashes -r /wheels/lock.txt \
 && rm -rf /wheels \
 && python -m pip freeze > /etc/k6-oracle-freeze.txt
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PIP_NO_INDEX=1
