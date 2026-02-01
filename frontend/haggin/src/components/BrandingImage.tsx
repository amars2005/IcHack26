import React from 'react';

export default function BrandingImage() {
  const [svgHtml, setSvgHtml] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    let mounted = true;
    const candidates = ['/BrandingImage.svg', '/image-removebg-preview.svg', '/branding-traced.svg', '/branding.svg'];
    (async () => {
      try {
        for (const c of candidates) {
          try {
            const res = await fetch(c);
            if (!mounted) return;
            if (!res.ok) continue;
            const text = await res.text();
            // Parse the SVG and remove any full-coverage dark/background paths (e.g., fill="#0A0A0A" or white backgrounds)
            try {
              const parser = new DOMParser();
              const doc = parser.parseFromString(text, 'image/svg+xml');
              const paths = Array.from(doc.querySelectorAll('path'));
              for (const p of paths) {
                const fill = (p.getAttribute('fill') || '').trim();
                // remove obvious solid background rectangles (very dark or very light fills)
                if (/^(#?0a0a0a|#?000000|black)$/i.test(fill) || /^(#?ffffff|#?fff|white)$/i.test(fill)) {
                  p.remove();
                }
              }
              const serializer = new XMLSerializer();
              const cleaned = serializer.serializeToString(doc.documentElement);
              setSvgHtml(cleaned);
              setLoading(false);
              return;
            } catch (err) {
              // parsing failed — fallback to raw image src
              setSvgHtml(null);
              setLoading(false);
              return;
            }
          } catch (e) {
            // try next candidate
          }
        }
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => { mounted = false; };
  }, []);

  if (loading) return <div style={{ height: 72, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>Loading…</div>;

  if (svgHtml) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', paddingBottom: 8, flexShrink: 0 }} dangerouslySetInnerHTML={{ __html: svgHtml }} />
    );
  }

  return (
    <div style={{ width: '100%', height: 72, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(255,255,255,0.02)', borderRadius: 8, color: '#94a3b8', fontSize: 13, flexShrink: 0 }}>
      No branding set
    </div>
  );
}
