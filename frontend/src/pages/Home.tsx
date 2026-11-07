import { Link } from 'react-router-dom';

function ModeCard({ to, mode, title, text }: { to: string; mode: string; title: string; text: string }) {
  return (
    <Link to={to} className="card group flex flex-col gap-3 transition-colors hover:border-accent">
      <span className="badge self-start">{mode}</span>
      <h2 className="text-3xl font-bold">{title}</h2>
      <p className="text-lg text-muted">{text}</p>
      <span className="mt-auto pt-2 text-lg font-semibold text-accent group-hover:underline">Open →</span>
    </Link>
  );
}

export default function Home() {
  return (
    <div className="mx-auto max-w-4xl py-6 sm:py-12">
      <h1 className="text-4xl font-bold leading-tight tracking-tight sm:text-6xl">
        Say a sentence without making a sound, <span className="text-accent">in your own voice.</span>
      </h1>
      <p className="mt-6 max-w-3xl text-xl text-muted">
        Most people with ALS eventually lose usable speech. Alex is still speaking today, but not for long. Small motion sensors on his chin and head
        pick up the words he silently mouths, and a clone of his own voice says them out loud.
      </p>
      <div className="mt-10 grid gap-5 md:grid-cols-2">
        <ModeCard to="/learn" mode="Mode 1" title="Learn" text="While Alex can still talk, his words automatically label his sensor data and bank his voice." />
        <ModeCard to="/silent-speech" mode="Mode 2" title="Silent Speech" text="Mouth a few words silently and hear a full sentence spoken in Alex's voice." />
      </div>
    </div>
  );
}
