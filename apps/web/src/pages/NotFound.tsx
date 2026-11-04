import { Link } from "react-router-dom";

export default function NotFound() {
  return (
    <section className="state" style={{ marginTop: 40 }}>
      <h1>Page not found</h1>
      <p>
        <Link to="/">Return to the start</Link>
      </p>
    </section>
  );
}
