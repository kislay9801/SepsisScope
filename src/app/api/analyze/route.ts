import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 60;

/**
 * In local development this route proxies to the Flask server on port 5000
 * (started automatically by `npm run dev` via concurrently).
 *
 * On Vercel this route is bypassed by the vercel.json rewrite which sends
 * /api/analyze directly to the api/analyze.py Python serverless function.
 */
const FLASK_URL =
  process.env.FLASK_API_URL ?? "http://localhost:5000/api/analyze";

export async function POST(request: NextRequest) {
  try {
    // Forward the raw multipart body to Flask
    const formData = await request.formData();

    const flaskResponse = await fetch(FLASK_URL, {
      method: "POST",
      body: formData,
      // Deliberately no Content-Type header — fetch sets boundary automatically
    });

    if (!flaskResponse.ok) {
      const text = await flaskResponse.text().catch(() => "Unknown error");
      return NextResponse.json(
        {
          status: "error",
          error: `Analysis server returned ${flaskResponse.status}: ${text}`,
        },
        { status: flaskResponse.status }
      );
    }

    const data = await flaskResponse.json();
    return NextResponse.json(data);
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);

    // Friendly error when Flask server isn't running yet
    const isConnectionRefused =
      message.includes("ECONNREFUSED") || message.includes("fetch failed");

    return NextResponse.json(
      {
        status: "error",
        error: isConnectionRefused
          ? "Python analysis server is not running. Start it with: python3 api/analyze.py"
          : `Proxy error: ${message}`,
      },
      { status: 502 }
    );
  }
}

export async function GET() {
  return NextResponse.json({ status: "ok", endpoint: FLASK_URL });
}
