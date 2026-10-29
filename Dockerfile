# Open House on a server (Hugging Face Spaces, or any host that runs Docker).
# The Jev key comes from the host's secrets as TYPESAFE_API_KEY; it is never in the image.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OPEN_HOUSE_HOSTED=1 \
    HF_HOME=/app/.cache/hf FASTEMBED_CACHE_PATH=/app/.cache/fastembed FEEDBACK_DIR=/app/feedback

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements-server.txt .
RUN pip install -r requirements-server.txt && python -m spacy download en_core_web_sm

COPY --chown=user . .
RUN mkdir -p /app/.cache /app/feedback && chown -R user /app
USER user

# Build the search indexes now, so the first visitor doesn't wait for them.
RUN python -c "from realtor.lease.sources import _index; [_index(k) for k in ('rta', 'lease', 'guide', 'case')]"

EXPOSE 7860
# No access log: visitors' addresses aren't kept.
CMD ["uvicorn", "realtor.web.app:app", "--host", "0.0.0.0", "--port", "7860", "--no-access-log", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
