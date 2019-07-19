import {NgModule} from '@angular/core';
import {RouterModule, Routes} from '@angular/router';
import {DagsComponent} from "./dags/dags.component";
import {DagComponent} from "./dag/dag.component";

const routes: Routes = [
    {
        path: '',
        component: DagComponent,
        children: [
            {
                path: 'dag-detail/:id',
                loadChildren:
                    './dag-detail/dag-detail.module#DagDetailModule'
            },
            {path: '', component: DagsComponent}
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
export class DagRoutingModule {
}


/*
Copyright Google LLC. All Rights Reserved.
Use of this source code is governed by an MIT-style license that
can be found in the LICENSE file at http://angular.io/license
*/