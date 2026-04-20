# Contributing

Contributions are welcome.

## Branch strategy

- `main`: stable branch for releases.
- `develop`: integration branch for ongoing work.
- `feature/<short-name>`: new features and improvements.
- `fix/<short-name>`: bug fixes.
- `docs/<short-name>`: documentation-only changes.

Please open pull requests against `develop`.

## Development setup

1. Create and activate a Python virtual environment.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Run the app:
   `uvicorn app.main:app --reload`

Or run with Docker:
`docker compose up --build`

## Pull requests

Please keep pull requests focused and include:

- Clear description of what changed and why
- Reproduction/validation steps
- UI screenshots or short clips for visual changes
- Related issue number when applicable

## Code style

- Keep changes small and pragmatic
- Preserve existing naming patterns
- Prefer explicit behavior over hidden magic

## Suggested workflow

1. Sync `develop`.
2. Create a branch from `develop`:
   `git checkout -b feature/my-change`
3. Commit with clear messages.
4. Push branch and open PR to `develop`.
5. After review and tests, maintainers merge into `develop`.
6. Release branches/tags are cut from `main`.
