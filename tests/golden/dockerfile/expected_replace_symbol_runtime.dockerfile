FROM python:3.12 AS builder

RUN pip install requests

FROM alpine:3.19 AS runtime

COPY --from=builder /app /app

CMD ["python"]
