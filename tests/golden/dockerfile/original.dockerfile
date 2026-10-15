FROM python:3.12 AS builder

RUN pip install requests

FROM alpine AS runtime

COPY --from=builder /app /app

CMD ["python"]
