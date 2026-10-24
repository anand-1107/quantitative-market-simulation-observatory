import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode } from 'react'
import { LoaderCircle } from 'lucide-react'

export function Panel({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <section className={`panel ${className}`}>{children}</section>
}

export function PanelTitle({ title, subtitle, action }: { title: string; subtitle?: string; action?: ReactNode }) {
  return <div className="panel-title"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>{action}</div>
}

export function Button({ children, busy, variant = 'primary', className = '', ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { busy?: boolean; variant?: 'primary' | 'secondary' | 'quiet' }) {
  return <button className={`button button-${variant} ${className}`} disabled={props.disabled || busy} {...props}>
    {busy ? <LoaderCircle size={15} className="spin" /> : null}{children}
  </button>
}

export function Field({ label, hint, ...props }: InputHTMLAttributes<HTMLInputElement> & { label: string; hint?: string }) {
  return <label className="field"><span>{label}</span><input {...props} />{hint && <small>{hint}</small>}</label>
}

export function SelectField({ label, children, ...props }: React.SelectHTMLAttributes<HTMLSelectElement> & { label: string; children: ReactNode }) {
  return <label className="field"><span>{label}</span><select {...props}>{children}</select></label>
}

export function Stat({ label, value, detail, tone }: { label: string; value: string; detail?: string; tone?: 'positive' | 'negative' | 'cyan' }) {
  return <div className={`stat ${tone ? `tone-${tone}` : ''}`}><span>{label}</span><strong>{value}</strong>{detail && <small>{detail}</small>}</div>
}

export function EmptyState({ icon, title, children }: { icon?: ReactNode; title: string; children: ReactNode }) {
  return <div className="empty-state">{icon && <div className="empty-icon">{icon}</div>}<strong>{title}</strong><p>{children}</p></div>
}

export function ErrorNotice({ children }: { children: ReactNode }) {
  return <div role="alert" className="notice notice-error">{children}</div>
}

export function SuccessNotice({ children }: { children: ReactNode }) {
  return <div role="status" className="notice notice-success">{children}</div>
}

export function PageHeading({ eyebrow, title, description, aside }: { eyebrow: string; title: string; description: string; aside?: ReactNode }) {
  return <div className="page-heading"><div><span className="eyebrow">{eyebrow}</span><h1>{title}</h1><p>{description}</p></div>{aside}</div>
}
