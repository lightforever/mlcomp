import {NgModule} from '@angular/core';
import {RouterModule, Routes} from '@angular/router';
import {SkynetComponent} from "./skynet/skynet.component";
import {SpaceComponent} from "./space/space.component";
import {MemoryComponent} from "./memory/memory.component";

const routes: Routes = [
    {
        path: '',
        component: SkynetComponent,
        children: [
                {path: 'space', component: SpaceComponent},
                {path: 'memory', component: MemoryComponent},
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
export class SkynetRoutingModule {
}
