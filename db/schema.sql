-- Step 4.1: the two tables of the monitor. Run this once.
-- Supabase: open your project, SQL Editor, New query, paste all of this, press Run.
-- It is safe to run twice (every line says "if not exists").

create table if not exists requests (
  id            bigint generated always as identity primary key,
  created_at    timestamptz not null default now(),
  source        text not null default 'app',          -- app (the public page) or eval (a test run)
  session       text,                                  -- a random id for one visit, not a person
  role          text not null default 'guest',
  mode          text not null default 'plain',         -- plain or agent
  question      text,                                  -- after the input guard, personal data hidden
  answer        text,                                  -- after the output guard, never the raw reply
  citations     jsonb not null default '[]'::jsonb,    -- [{"doc": "...", "page": 60}]
  steps         jsonb not null default '[]'::jsonb,    -- the agent tool steps
  found         boolean,
  tokens_in     integer not null default 0,
  tokens_out    integer not null default 0,
  cost_cents    numeric(12,5) not null default 0,
  seconds       numeric(9,3) not null default 0,
  blocked       text not null default '',              -- why the answer was replaced, empty when not
  pii_hidden    boolean not null default false,        -- personal data was hidden from Claude
  injection     boolean not null default false,        -- an instruction was found inside a piece
  no_citation   boolean not null default false,        -- the bot said it found the answer but cited nothing
  thumb         smallint,                              -- 1 up, -1 down, empty when not rated
  review_note   text,                                  -- your note from the weekly review
  failure_type  text                                   -- your failure type from the weekly review
);

create index if not exists requests_created_idx on requests (created_at desc);
create index if not exists requests_thumb_idx on requests (thumb) where thumb is not null;

create table if not exists eval_runs (
  id          bigint generated always as identity primary key,
  created_at  timestamptz not null default now(),
  tag         text not null,
  split       text,
  partial     boolean not null default false,          -- true when only some questions were run
  questions   integer not null default 0,
  settings    jsonb not null default '{}'::jsonb,
  scores      jsonb not null default '{}'::jsonb,      -- {"correct": {"good": 61, "of": 75, "points": 81.3}, ...}
  cost_cents  numeric(12,5) not null default 0
);

create index if not exists eval_runs_created_idx on eval_runs (created_at desc);

-- Supabase may show a warning that these two tables are open. That is expected for this demo:
-- the app is the only user, and it talks to the tables from the server, not from the page.
