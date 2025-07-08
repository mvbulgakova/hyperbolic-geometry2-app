import plotly.graph_objects as go
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# ==============================================================================
# 1. ФУНКЦИЯ ДЛЯ СОЗДАНИЯ ГРАФИКА ОРИСФЕРЫ
# ==============================================================================
def create_orosphere_figure(phi_deg, theta_deg, k_prime, show_guiding_lines=True, current_camera=None):
    r = 1.0  # Радиус сферы-абсолюта
    
    phi_rad = np.deg2rad(phi_deg)
    theta_rad = np.deg2rad(theta_deg)

    # Точка касания орисферы на Абсолюте.
    touch_point = np.array([
        r * np.cos(phi_rad) * np.sin(theta_rad),
        r * np.sin(phi_rad) * np.sin(theta_rad),
        r * np.cos(theta_rad)
    ])
    
    # --- РАСЧЕТ ЭЛЛИПСОИДА ДЛЯ МОДЕЛИ БЕЛЬТРАМИ-КЛЕЙНА (ИСПРАВЛЕННЫЙ) ---
    # k_prime - внутренний математический параметр. Больший k_prime = большая/глубокая орисфера.
    if k_prime < 1e-6: k_prime = 1e-6

    # Канонические формулы для параметров эллипсоида, теперь правильные
    center_ellipsoid = touch_point * (1 / (1 + k_prime))
    radius_parallel = k_prime / (1 + k_prime)
    radius_perp = np.sqrt(k_prime) / np.sqrt(1 + k_prime) # <-- ВОТ ЗДЕСЬ БЫЛА ОШИБКА


    fig = go.Figure()

    # Сфера-абсолют
    phi_surf = np.linspace(0, 2*np.pi, 50)
    theta_surf = np.linspace(0, np.pi, 50)
    x_abs = r * np.outer(np.cos(phi_surf), np.sin(theta_surf))
    y_abs = r * np.outer(np.sin(phi_surf), np.sin(theta_surf))
    z_abs = r * np.outer(np.ones_like(phi_surf), np.cos(theta_surf))
    fig.add_trace(go.Surface(x=x_abs, y=y_abs, z=z_abs, colorscale='Blues', opacity=0.15, showscale=False, name='Абсолют', hoverinfo='none'))

    # Орисфера (Эллипсоид)
    x_unit_sphere = np.outer(np.cos(phi_surf), np.sin(theta_surf))
    y_unit_sphere = np.outer(np.sin(phi_surf), np.sin(theta_surf))
    z_unit_sphere = np.outer(np.ones_like(phi_surf), np.cos(theta_surf))
    
    x_ell_std = radius_perp * x_unit_sphere
    y_ell_std = radius_perp * y_unit_sphere
    z_ell_std = radius_parallel * z_unit_sphere

    u_z = np.array([0., 0., 1.])
    u_z_prime = touch_point
    
    v = np.cross(u_z, u_z_prime)
    s = np.linalg.norm(v)
    c = np.dot(u_z, u_z_prime)

    if s < 1e-9:
        R = np.sign(c) * np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))

    coords = np.vstack([x_ell_std.ravel(), y_ell_std.ravel(), z_ell_std.ravel()])
    rotated_coords = R @ coords
    
    x_horo = rotated_coords[0, :].reshape(x_ell_std.shape) + center_ellipsoid[0]
    y_horo = rotated_coords[1, :].reshape(y_ell_std.shape) + center_ellipsoid[1]
    z_horo = rotated_coords[2, :].reshape(z_ell_std.shape) + center_ellipsoid[2]
    
    fig.add_trace(go.Surface(x=x_horo, y=y_horo, z=z_horo, colorscale='Greens', opacity=0.4, showscale=False, name='Орисфера', hoverinfo='none'))

    # Маркер для точки касания
    fig.add_trace(go.Scatter3d(x=[touch_point[0]], y=[touch_point[1]], z=[touch_point[2]],
                               mode='markers', marker=dict(color='red', size=5, symbol='circle'),
                               name='Точка касания', showlegend=False, hoverinfo='none'))

    # --- ГЕОДЕЗИЧЕСКИЕ (ПУЧОК ПРЯМЫХ) ---
    num_lines_total = 50
    indices = np.arange(0, num_lines_total, dtype=float) + 0.5
    phi_end_vals_gs = np.arccos(1 - 2 * indices / num_lines_total) 
    theta_end_vals_gs = np.pi * (1 + 5**0.5) * indices 
    
    for i in range(num_lines_total):
        p_start = touch_point
        p_end = np.array([
            r * np.cos(theta_end_vals_gs[i]) * np.sin(phi_end_vals_gs[i]),
            r * np.sin(theta_end_vals_gs[i]) * np.sin(phi_end_vals_gs[i]),
            r * np.cos(phi_end_vals_gs[i])
        ])
            
        if np.linalg.norm(p_end - p_start) < 1e-6: continue
        vec_line = p_end - p_start
        
        if radius_perp < 1e-6 or radius_parallel < 1e-6: continue
            
        D_inv = np.diag([1/radius_perp**2, 1/radius_perp**2, 1/radius_parallel**2])
        M_inv = R @ D_inv @ R.T

        oc_to_line_start = p_start - center_ellipsoid
        
        a_ell = vec_line.T @ M_inv @ vec_line
        b_ell = 2 * (vec_line.T @ M_inv @ oc_to_line_start)
        c_ell = oc_to_line_start.T @ M_inv @ oc_to_line_start - 1

        discriminant = b_ell**2 - 4 * a_ell * c_ell

        if discriminant < 1e-7:
            fig.add_trace(go.Scatter3d(x=[p_start[0], p_end[0]], y=[p_start[1], p_end[1]], z=[p_start[2], p_end[2]],
                                       mode='lines', line=dict(color='red', width=2), showlegend=False, hoverinfo='none'))
            continue

        t1 = (-b_ell - np.sqrt(discriminant)) / (2*a_ell)
        t2 = (-b_ell + np.sqrt(discriminant)) / (2*a_ell)
        
        valid_t = sorted([t for t in [t1,t2] if 0-1e-6 <= t <= 1+1e-6])

        if len(valid_t) == 2:
            t_entry, t_exit = valid_t
            p_entry = p_start + t_entry * vec_line
            p_exit = p_start + t_exit * vec_line

            if t_entry > 1e-6:
                fig.add_trace(go.Scatter3d(x=[p_start[0], p_entry[0]], y=[p_start[1], p_entry[1]], z=[p_start[2], p_entry[2]], mode='lines', line=dict(color='red', width=3), showlegend=False, hoverinfo='none'))
            
            fig.add_trace(go.Scatter3d(x=[p_entry[0], p_exit[0]], y=[p_entry[1], p_exit[1]], z=[p_entry[2], p_exit[2]], mode='lines', line=dict(color='red', width=2, dash='dash'), showlegend=False, hoverinfo='none'))

            if 1 - t_exit > 1e-6:
                fig.add_trace(go.Scatter3d(x=[p_exit[0], p_end[0]], y=[p_exit[1], p_end[1]], z=[p_exit[2], p_end[2]], mode='lines', line=dict(color='red', width=3), showlegend=False, hoverinfo='none'))
        else:
            fig.add_trace(go.Scatter3d(x=[p_start[0], p_end[0]], y=[p_start[1], p_end[1]], z=[p_start[2], p_end[2]], mode='lines', line=dict(color='red', width=2), showlegend=False, hoverinfo='none'))

    if show_guiding_lines:
        t_eq = np.linspace(0, 2 * np.pi, 100)
        x_eq, y_eq, z_eq = r * np.cos(t_eq), r * np.sin(t_eq), np.zeros_like(t_eq)
        fig.add_trace(go.Scatter3d(x=x_eq, y=y_eq, z=z_eq, mode='lines', line=dict(color='purple', width=2), name='Экватор (для Phi)', showlegend=True))
        
        meridian_phi_rad = phi_rad 
        t_mer = np.linspace(0, np.pi, 100)
        x_mer, y_mer, z_mer = r * np.cos(meridian_phi_rad) * np.sin(t_mer), r * np.sin(meridian_phi_rad) * np.sin(t_mer), r * np.cos(t_mer)
        fig.add_trace(go.Scatter3d(x=x_mer, y=y_mer, z=z_mer, mode='lines', line=dict(color='orange', width=2), name='Меридиан (для Theta)', showlegend=True))
        
    scene_settings = dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data')
    
    initial_camera_eye = dict(x=1.5, y=1.5, z=1.5) 
    if current_camera: scene_settings['camera'] = current_camera
    else: scene_settings['camera_eye'] = initial_camera_eye

    fig.update_layout(
        title='Интерактивная модель орисферы в модели Клейна (исправлено)',
        scene=scene_settings, margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0.01, y=0.99),
        font=dict(family="Arial, sans-serif", size=12, color="black")
    )
    return fig

# ==============================================================================
# 2. СОЗДАНИЕ ПРИЛОЖЕНИЯ DASH
# ==============================================================================
app = dash.Dash(__name__)

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'fontSize': '12px', 'color': 'black'}, children=[
    html.H1("Интерактивная модель Бельтрами-Клейна", style={'textAlign': 'center'}),
    
    html.Div(style={'display': 'flex', 'justifyContent': 'center', 'margin-bottom': '20px'}, children=[
        html.Button('Скрыть/показать направляющие', id='toggle-guiding-lines-button', n_clicks=0, style={'width': 'auto', 'padding': '10px 20px'}),
        dcc.Store(id='guiding-lines-visibility-store', data={'visible': True})
    ]),

    html.Div(style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'center', 'alignItems': 'flex-start', 'gap': '20px'}, children=[
        
        dcc.Graph(id='hyperbolic-orosphere-graph', style={'flexGrow': '1', 'height': '70vh', 'width': 'auto'}),

        html.Div(style={'flexShrink': '0', 'width': '300px', 'padding': '20px', 'border': '1px solid #ccc', 'borderRadius': '8px'}, children=[
            html.Label("Долгота (phi) точки касания", style={'margin-top': '0px', 'display': 'block'}),
            dcc.Slider(id='phi-slider', min=0, max=360, step=1, value=45, marks={i: str(i) for i in range(0, 361, 45)}),
            
            html.Label("Широта (theta) точки касания", style={'margin-top': '10px', 'display': 'block'}),
            dcc.Slider(id='theta-slider', min=0, max=180, step=1, value=90, marks={i: str(i) for i in range(0, 181, 30)}),

            html.Label("Размер орисферы (параметр k')", style={'margin-top': '10px', 'display': 'block'}),
            dcc.Slider(id='r-horo-slider', min=0.1, max=10.0, step=0.1, value=3.0, marks={i: str(i) for i in range(0, 11)}),
        ])
    ])
])

# ==============================================================================
# 3. ЛОГИКА ОБНОВЛЕНИЯ ГРАФИКА
# ==============================================================================
@app.callback(
    Output('hyperbolic-orosphere-graph', 'figure'),
    Output('guiding-lines-visibility-store', 'data'),
    [Input('phi-slider', 'value'),
     Input('theta-slider', 'value'),
     Input('r-horo-slider', 'value'),
     Input('toggle-guiding-lines-button', 'n_clicks')],
    [State('hyperbolic-orosphere-graph', 'relayoutData'),
     State('guiding-lines-visibility-store', 'data')]
)
def update_figure(phi, theta, r_horo, n_clicks, relayoutData, current_visibility_data):
    current_camera = None
    if relayoutData and 'scene.camera' in relayoutData:
        current_camera = relayoutData['scene.camera']

    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'no_trigger'

    new_visibility_data = current_visibility_data
    if button_id == 'toggle-guiding-lines-button':
        new_visibility_state = not current_visibility_data['visible']
        new_visibility_data = {'visible': new_visibility_state}
    
    fig = create_orosphere_figure(
        phi, theta, r_horo, 
        show_guiding_lines=new_visibility_data['visible'],
        current_camera=current_camera
    )
    
    return fig, new_visibility_data

if __name__ == '__main__':
    app.run(debug=True)