import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { EnterpriseWorkspace } from './EnterpriseWorkspace'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <EnterpriseWorkspace />
  </StrictMode>,
)
