import {Component, OnInit, ViewChild, EventEmitter} from '@angular/core';
import {Dag, PaginatorRes, Project} from '../models';
import {ProjectService} from '../project.service';
import {MatSort, MatTableDataSource, MatPaginator} from '@angular/material';
import {Observable, of as observableOf, merge} from 'rxjs';
import {catchError} from 'rxjs/operators';
import {map} from 'rxjs/operators';
import {startWith} from 'rxjs/operators';
import {switchMap} from 'rxjs/operators';

@Component({
    selector: 'app-project',
    templateUrl: './project.component.html',
    styleUrls: ['./project.component.css']
})
export class ProjectComponent implements OnInit {
    dataSource: MatTableDataSource<Project> = new MatTableDataSource();

    @ViewChild(MatPaginator) paginator: MatPaginator;
    @ViewChild(MatSort) sort: MatSort;
    change: EventEmitter<any> = new EventEmitter();

    displayed_columns: string[] = ['name', 'dag_count', 'last_activity'];
    isLoading_results = false;
    total: number;
    private interval: number;

    constructor(private project_service: ProjectService) {
    }

    ngOnInit() {
        // If the user changes the sort order, reset back to the first page.
        this.sort.sortChange.subscribe(() => this.paginator.pageIndex = 0);

        merge(this.sort.sortChange, this.paginator.page, this.change)
            .pipe(
                startWith({}),
                switchMap(() => {
                    this.isLoading_results = true;
                    return this.project_service.getProjects(
                        this.sort.active ? this.sort.active : '',
                        this.sort.direction ? this.sort.direction == 'desc' : true,
                        this.paginator.pageIndex,
                        this.paginator.pageSize||10,
                        this.dataSource.filter
                    );
                }),
                map(res => {
                    // Flip flag to show that loading has finished.
                    this.isLoading_results = false;

                    return res;
                }),
                catchError(() => {
                    this.isLoading_results = false;
                    return observableOf(new PaginatorRes<Project>());
                })
            ).subscribe(res => {this.dataSource.data = res.data; this.total = res.total });

        this.interval = setInterval(() => this.change.emit('event'), 5000);
    }

    ngOnDestroy(){
        clearInterval(this.interval);
    }

    applyFilter(filterValue: string) {
        this.dataSource.filter = filterValue;

        if (this.dataSource.paginator) {
            this.dataSource.paginator.firstPage();
        }

        this.change.emit();
    }
}
