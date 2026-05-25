# Deployment

## Recommended: Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io/.
3. Create a new app from the repository.
4. Set the main file path:

   ```text
   app/streamlit_app.py
   ```

5. Add this secret in Streamlit Cloud app settings:

   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```

6. Deploy.

Use the app with:

- `LLM provider`: `claude-api`
- `생성 방식`: `direct-draft`
- `Claude model`: `auto`

## Docker Deploy

Any Docker host can run:

```bash
docker build -t kolon-industry-mvp .
docker run --rm -p 8501:8501 -e ANTHROPIC_API_KEY="sk-ant-..." kolon-industry-mvp
```

Then open:

```text
http://localhost:8501
```

## Notes

- Do not commit `.streamlit/secrets.toml`.
- Do not commit raw IR decks, deal memos, transcripts, or internal report samples.
- For best output quality, use `direct-draft`. Use `pipeline` only when demonstrating extraction/KG/evidence architecture.
