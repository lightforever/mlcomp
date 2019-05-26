import {NgModule} from '@angular/core';
import {RouterModule, Routes} from '@angular/router';

import {CodeComponent} from './code/code.component';
import {ConfigComponent} from './config/config.component';
import {GraphComponent} from './graph/graph.component';
import {DagDetailComponent} from './dag-detail/dag-detail.component';
import {TasksComponent} from "../../task/tasks/tasks.component"


const routes: Routes = [
    {

        path: '',
        component: DagDetailComponent,
        children: [
            {path: 'code', component: CodeComponent},
            {path: 'config', component: ConfigComponent},
            {path: 'graph', component: GraphComponent},
            {path: '', component: TasksComponent}
        ]


    }
];

@NgModule({
    imports: [
        RouterModule.forChild(routes)
    ],
    exports: [
        RouterModule
    ]
})
export class DagDetailRoutingModule {
}